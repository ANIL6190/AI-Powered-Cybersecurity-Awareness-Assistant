"""Tests for the Phase 1 prepare stage (src/data/make_dataset.py) and URL helpers."""

import os

import pandas as pd
import pytest
import yaml

from src.data import make_dataset
from src.utils.urls import canonicalize_url, has_path, host_only, registrable_domain


def _read(cfg, name):
    return pd.read_csv(os.path.join(cfg["output"]["processed_dir"], f"{name}.csv"), keep_default_na=False,
                       dtype={"event_date": str})


# ------------------------------------------------------------------ URL helpers
@pytest.mark.parametrize("raw, expected", [
    ("https://www.Example.com", "example.com"),
    ("http://example.com/", "example.com"),
    ("example.com", "example.com"),
    ("HTTP://WWW.EXAMPLE.COM/Login/Verify.php?id=1", "example.com/Login/Verify.php?id=1"),
    ("https://www2.bank.co.uk/", "bank.co.uk"),
])
def test_canonicalize_removes_collection_format(raw, expected):
    assert canonicalize_url(raw) == expected


def test_url_text_is_preserved_not_collapsed():
    # Regression for the legacy bug where every URL became the single token "url".
    a = canonicalize_url("https://www.google.com")
    b = canonicalize_url("http://paypal-verify-login.xyz/secure/update.php")
    assert a != b
    assert "paypal-verify-login.xyz" in b and "/secure/update.php" in b


def test_registrable_domain_and_private_suffixes():
    assert registrable_domain("a.b.example.co.uk") == "example.co.uk"
    assert registrable_domain("x.web.app", include_private=False) == "web.app"
    assert registrable_domain("x.web.app", include_private=True) == "x.web.app"
    assert registrable_domain("192.168.1.1") == "192.168.1.1"


def test_has_path_and_host_only():
    assert has_path("example.com/login")
    assert not has_path("example.com")
    assert host_only("example.com/login?x=1") == "example.com"


# ------------------------------------------------------------------ prepare stage
def test_prepare_writes_all_outputs(prepared_project):
    _, cfg, _ = prepared_project
    for name in ("train", "val", "test", "shift_test", "temporal_test"):
        assert os.path.exists(os.path.join(cfg["output"]["processed_dir"], f"{name}.csv"))
    assert os.path.exists(os.path.join(cfg["output"]["reports_dir"], "prepare_summary.json"))


def test_labels_are_binary_and_phiusiil_direction_is_inverted(prepared_project):
    _, cfg, _ = prepared_project
    pool = pd.concat([_read(cfg, s) for s in ("train", "val", "test")])
    assert set(pool["label"]) == {0, 1}
    legit = pool[pool["model_text"] == "legit-site1.com"]
    assert legit["label"].iloc[0] == 0 and legit["label_name"].iloc[0] == "legitimate"
    phish = pool[pool["model_text"].str.startswith("secure-login1.xyz")]
    assert phish["label"].iloc[0] == 1


def test_conflicts_and_duplicates_removed(prepared_project):
    _, cfg, summary = prepared_project
    pool = pd.concat([_read(cfg, s) for s in ("train", "val", "test")])
    assert "conflict-example.com" not in set(pool["model_text"])
    assert summary["phiusiil"]["label_conflict_texts"] == 1
    assert summary["phiusiil"]["duplicate_rows_dropped"] >= 1
    assert not pool["model_text"].duplicated().any()


def test_split_is_domain_disjoint(prepared_project):
    _, cfg, _ = prepared_project
    domains = {s: set(_read(cfg, s)["registrable_domain"]) for s in ("train", "val", "test")}
    assert not domains["train"] & domains["val"]
    assert not domains["train"] & domains["test"]
    assert not domains["val"] & domains["test"]


def test_group_split_is_deterministic():
    df = pd.DataFrame({"registrable_domain": [f"d{i}.com" for i in range(500)]})
    a = make_dataset.assign_group_split(df, 0.7, 0.15, 0.15, seed=42)
    b = make_dataset.assign_group_split(df, 0.7, 0.15, 0.15, seed=42)
    assert (a == b).all()
    assert 0.6 < (a == "train").mean() < 0.8


def test_training_pool_has_no_dates_and_no_phishvn(prepared_project):
    _, cfg, _ = prepared_project
    pool = pd.concat([_read(cfg, s) for s in ("train", "val", "test")])
    assert set(pool["dataset"]) == {"phiusiil"}
    assert (pool["event_date"] == "").all()


def test_temporal_set_only_attested_dates_after_cutoff(prepared_project):
    _, cfg, summary = prepared_project
    temporal = _read(cfg, "temporal_test")
    dates = pd.to_datetime(temporal["event_date"], format="%Y-%m-%d")
    assert (dates > pd.Timestamp(cfg["temporal"]["cutoff_date"])).all()
    assert set(temporal["source"]) <= {"tinnhiemmang", "tinnhiem_org"}
    assert summary["temporal_test_labels"] == {"legitimate": 6, "phishing": 10}


def test_no_timestamp_is_imputed(prepared_project):
    _, cfg, _ = prepared_project
    shift = _read(cfg, "shift_test")
    # Invalid attested date stays undated and therefore is not in temporal_test
    bad = shift[shift["model_text"] == "bad-date-scam.com"]
    assert len(bad) == 1 and bad["event_date"].iloc[0] == ""
    assert "bad-date-scam.com" not in set(_read(cfg, "temporal_test")["model_text"])
    # Undated sources never receive a date
    undated = shift[shift["source"].isin(["tinnhiem_web", "tranco_vn", "tranco"])]
    assert (undated["event_date"] == "").all()


def test_bronze_tier_excluded(prepared_project):
    _, cfg, _ = prepared_project
    shift = _read(cfg, "shift_test")
    assert "bronze-scam.com" not in set(shift["model_text"])
    assert set(shift["tier"]) <= {"gold", "silver"}


def test_eval_rows_overlapping_training_pool_are_removed(prepared_project):
    _, cfg, summary = prepared_project
    shift = _read(cfg, "shift_test")
    assert "legit-site5.com" not in set(shift["model_text"])
    assert summary["phishvn"]["removed_exact_text_overlap"] + summary["phishvn"]["removed_domain_overlap"] >= 1


def test_include_sms_true_is_rejected(synthetic_project, tmp_path):
    config_path, cfg = synthetic_project
    cfg["task"]["include_sms"] = True
    bad_path = tmp_path / "params_sms.yaml"
    with open(bad_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    with pytest.raises(ValueError, match="include_sms"):
        make_dataset.run(str(bad_path))


def test_attested_date_parsing_is_strict():
    parsed, unparseable = make_dataset.parse_attested_dates(
        pd.Series(["18/02/2025", "", "2024-12-01", "31/02/2024"]), "%d/%m/%Y")
    assert parsed.iloc[0] == pd.Timestamp("2025-02-18")
    assert parsed.iloc[1:].isna().all()
    assert unparseable == 2
