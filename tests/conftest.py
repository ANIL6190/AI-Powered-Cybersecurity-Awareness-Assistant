"""
Shared fixtures for the Phase 1 tests.

All tests run on small synthetic data written to a temporary directory; they never
touch data/raw or the real processed outputs.
"""

import copy
import os

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF = "2024-03-08"


def _phiusiil_rows():
    rows = []
    for i in range(120):
        rows.append({"URL": f"https://www.legit-site{i}.com", "label": 1})
    for i in range(80):
        rows.append({"URL": f"http://secure-login{i}.xyz/verify/account.php", "label": 0})
    for i in range(30):
        rows.append({"URL": f"https://tenant{i}.web.app/", "label": 0})
    # Same canonical text with opposite labels -> must be dropped as a conflict
    rows.append({"URL": "http://www.conflict-example.com", "label": 0})
    rows.append({"URL": "https://www.conflict-example.com", "label": 1})
    # Exact duplicate
    rows.append({"URL": "https://www.legit-site0.com", "label": 1})
    return rows


def _phishvn_rows():
    def row(url, label, source, tier, collected_at=""):
        return {"channel": "url", "label": label, "source": source, "tier": tier, "collected_at": collected_at,
                "scraped_at": "2026-07-13T08:50:44+00:00", "url": url, "scenario": "other", "split": "train"}

    rows = []
    for i in range(12):  # NCSC phishing before the cutoff
        rows.append(row(f"vn-old-scam{i}.com", "phishing", "tinnhiemmang", "gold", "15/06/2022"))
    for i in range(10):  # NCSC phishing after the cutoff
        rows.append(row(f"vn-new-scam{i}.com", "phishing", "tinnhiemmang", "silver", "10/10/2024"))
    for i in range(6):  # certified organisations after the cutoff
        rows.append(row(f"https://agency{i}.gov.vn/", "benign", "tinnhiem_org", "gold", "01/05/2025"))
    for i in range(4):  # certified organisations before the cutoff
        rows.append(row(f"https://oldagency{i}.gov.vn/", "benign", "tinnhiem_org", "gold", "01/05/2021"))
    for i in range(8):  # undated benign
        rows.append(row(f"school{i}.edu.vn", "benign", "tinnhiem_web", "gold"))
        rows.append(row(f"popular{i}.vn", "benign", "tranco_vn", "silver"))
    # Bronze, reconstructed ISO date: excluded by tier filter
    rows.append(row("bronze-scam.com", "phishing", "chongluadao", "bronze", "2024-12-01"))
    # Invalid attested date: must stay undated (never imputed) and stay out of temporal_test
    rows.append(row("bad-date-scam.com", "phishing", "tinnhiemmang", "gold", "31/02/2024"))
    # Overlaps a PhiUSIIL registrable domain: must be removed from the evaluation sets
    rows.append(row("legit-site5.com", "benign", "tranco", "silver"))
    return rows


@pytest.fixture
def base_config():
    with open(os.path.join(ROOT, "params.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def synthetic_project(tmp_path, base_config):
    """Synthetic raw files plus a params.yaml pointing at tmp_path. Returns (config_path, cfg)."""
    raw = tmp_path / "raw"
    pvn_dir = raw / "phishvn"
    pvn_dir.mkdir(parents=True)
    pd.DataFrame(_phiusiil_rows()).to_csv(raw / "phiusiil.csv", index=False)
    pd.DataFrame(_phishvn_rows()).to_csv(pvn_dir / "dataset_url.csv", index=False)

    cfg = copy.deepcopy(base_config)
    cfg["raw_data"] = {
        "phiusiil_path": str(raw / "phiusiil.csv"),
        "phishvn_dir": str(pvn_dir),
        "phishvn_path": str(pvn_dir / "dataset_url.csv"),
        "sms_path": str(raw / "SMSSpamCollection"),
    }
    cfg["temporal"]["cutoff_date"] = CUTOFF
    cfg["validation"] = {"max_class_ratio_drift": 0.5, "min_temporal_rows": 5}
    cfg["features"]["tfidf"].update({"min_df": 1, "max_features": 5000})
    cfg["models"]["xgboost"].update({"n_estimators": 20, "early_stopping_rounds": 5, "n_jobs": 1})
    cfg["mlflow"].update({"default_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                          "local_artifact_root": str(tmp_path / "mlartifacts"), "log_models": False})
    cfg["output"] = {
        "processed_dir": str(tmp_path / "processed"),
        "features_dir": str(tmp_path / "processed" / "features"),
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
    }
    config_path = tmp_path / "params.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path), cfg


@pytest.fixture
def prepared_project(synthetic_project, monkeypatch):
    """synthetic_project after the prepare stage has run."""
    from src.data import make_dataset

    config_path, cfg = synthetic_project
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    summary = make_dataset.run(config_path)
    return config_path, cfg, summary
