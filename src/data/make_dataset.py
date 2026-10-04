"""
Phase 1 `prepare` stage: binary URL phishing dataset construction.

Inputs (DVC-tracked raw data):
  - PhiUSIIL Phishing URL Dataset  -> training pool + normal held-out test (undated)
  - PhishVN v3.1.0 dataset_url.csv  -> separate evaluation-only data:
        * shift_test:    all verified (gold/silver) PhishVN rows   (cross-source evaluation)
        * temporal_test: PhishVN rows whose *source-attested* event date is after temporal.cutoff_date,
                         a split point inside PhishVN (time-stratified evaluation; PhiUSIIL is undated,
                         so no claim is made that these rows are newer than the training data)
SMSSpamCollection is versioned for provenance but excluded from this task (task.include_sms=false).

Outputs (data/processed/phase1/): train.csv, val.csv, test.csv, shift_test.csv, temporal_test.csv
and reports/phase1/prepare_summary.json.

Labels: 1 = phishing, 0 = legitimate.
No timestamps are created or imputed: PhiUSIIL rows have no event_date, and PhishVN rows only get
one when the source itself attests it (see phishvn.attested_date_sources).
"""

import argparse
import hashlib
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.utils.config import ensure_dir, get_logger, load_config, set_seed, write_json
from src.utils.urls import (
    canonicalize_url,
    extract_host,
    has_path,
    registrable_domain,
    url_representation,
)

logger = get_logger("make_dataset")

OUTPUT_COLUMNS = [
    "model_text", "label", "label_name", "dataset", "source", "tier", "registrable_domain",
    "has_path", "event_date", "url_raw", "split",
]
LABEL_NAMES = {1: "phishing", 0: "legitimate"}


# ==============================================================================
# Loading
# ==============================================================================
def load_phiusiil(path: str) -> pd.DataFrame:
    """PhiUSIIL: label 1 = legitimate, 0 = phishing (inverted to 1 = phishing here)."""
    raw = pd.read_csv(path, usecols=["URL", "label"], encoding="utf-8-sig", low_memory=False)
    bad = ~raw["label"].isin([0, 1])
    if bad.any():
        raise ValueError(f"PhiUSIIL contains {int(bad.sum())} rows with labels outside {{0, 1}}.")
    return pd.DataFrame({
        "url_raw": raw["URL"].astype(str),
        "label": (1 - raw["label"].astype(int)),
        "dataset": "phiusiil",
        "source": "PhiUSIIL",
        "tier": "",
        "event_date": pd.NaT,
    })


def parse_attested_dates(values: pd.Series, date_format: str) -> Tuple[pd.Series, int]:
    """Strictly parse source dates; values that do not match the format stay NaT (never imputed)."""
    s = values.fillna("").astype(str).str.strip()
    present = ~s.isin(["", "nan", "NaN", "None"])
    parsed = pd.to_datetime(s.where(present), format=date_format, errors="coerce")
    unparseable = int((present & parsed.isna()).sum())
    return parsed, unparseable


def load_phishvn(path: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"url", "label", "source", "tier", cfg["timestamp_column"]}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"PhishVN file is missing expected columns: {sorted(missing)}")

    stats: Dict[str, Any] = {"rows_raw": int(len(raw))}
    raw = raw[raw["tier"].isin(cfg["tiers"])].copy()
    stats["rows_after_tier_filter"] = int(len(raw))
    stats["tiers_used"] = list(cfg["tiers"])

    label_map = {cfg["positive_value"]: 1, cfg["negative_value"]: 0}
    unknown = ~raw["label"].isin(label_map)
    if unknown.any():
        raise ValueError(f"PhishVN has unexpected label values: {sorted(raw.loc[unknown, 'label'].unique())}")

    # Event dates only from sources whose date is attested by the source itself.
    attested = raw["source"].isin(cfg["attested_date_sources"])
    parsed, unparseable = parse_attested_dates(raw.loc[attested, cfg["timestamp_column"]], cfg["timestamp_format"])
    event_date = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    event_date.loc[attested] = parsed
    stats["attested_date_sources"] = list(cfg["attested_date_sources"])
    stats["unparseable_attested_dates"] = unparseable
    stats["rows_with_event_date"] = int(event_date.notna().sum())

    df = pd.DataFrame({
        "url_raw": raw["url"].astype(str),
        "label": raw["label"].map(label_map).astype(int),
        "dataset": "phishvn",
        "source": raw["source"],
        "tier": raw["tier"],
        "event_date": event_date,
    }).reset_index(drop=True)
    return df, stats


# ==============================================================================
# URL processing
# ==============================================================================
def build_url_columns(df: pd.DataFrame, url_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    canonical = out["url_raw"].map(lambda u: canonicalize_url(
        u,
        strip_scheme=url_cfg.get("strip_scheme", True),
        strip_www=url_cfg.get("strip_www", True),
        strip_trailing_slash=url_cfg.get("strip_trailing_slash", True),
    ))
    hosts = canonical.map(extract_host)
    include_private = bool(url_cfg.get("include_psl_private_domains", False))
    out["registrable_domain"] = hosts.map(lambda h: registrable_domain(h, include_private))
    out["has_path"] = canonical.map(has_path).astype(int)
    out["model_text"] = canonical.map(lambda c: url_representation(c, url_cfg.get("representation", "canonical_url")))
    return out


def drop_invalid(df: pd.DataFrame, min_length: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    too_short = df["model_text"].str.len() < min_length
    no_domain = df["registrable_domain"].fillna("") == ""
    bad = too_short | no_domain
    stats = {"dropped_too_short": int(too_short.sum()), "dropped_no_domain": int((no_domain & ~too_short).sum())}
    return df[~bad].reset_index(drop=True), stats


def dedupe_and_resolve_conflicts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove duplicates on the model input. If the same model_text carries both labels
    (e.g. 'http://www.x.com' phishing vs 'https://www.x.com' legitimate after canonicalisation),
    every copy is dropped: there is no defensible label for it.
    """
    labels_per_text = df.groupby("model_text")["label"].nunique()
    conflicting = set(labels_per_text[labels_per_text > 1].index)
    conflict_mask = df["model_text"].isin(conflicting)
    clean = df[~conflict_mask]
    deduped = clean.drop_duplicates(subset=["model_text"], keep="first")
    stats = {
        "label_conflict_texts": int(len(conflicting)),
        "label_conflict_rows_dropped": int(conflict_mask.sum()),
        "duplicate_rows_dropped": int(len(clean) - len(deduped)),
    }
    return deduped.reset_index(drop=True), stats


# ==============================================================================
# Splitting
# ==============================================================================
def group_bucket(group: str, seed: int) -> float:
    """Deterministic, platform-independent position in [0, 1) for a group key."""
    digest = hashlib.md5(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


def assign_group_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    """Assign whole registrable-domain groups to train/val/test so no domain spans two splits."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    buckets = df["registrable_domain"].map(lambda g: group_bucket(g, seed))
    split = pd.Series("test", index=df.index)
    split[buckets < train_ratio + val_ratio] = "val"
    split[buckets < train_ratio] = "train"
    return split


# ==============================================================================
# Evaluation sets (PhishVN, evaluation only)
# ==============================================================================
def remove_overlap(eval_df: pd.DataFrame, pool_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop evaluation rows whose model_text or registrable domain occurs anywhere in the PhiUSIIL pool."""
    seen_text = set(pool_df["model_text"])
    seen_domain = set(pool_df["registrable_domain"])
    text_hit = eval_df["model_text"].isin(seen_text)
    domain_hit = eval_df["registrable_domain"].isin(seen_domain) & ~text_hit
    keep = ~(text_hit | domain_hit)
    stats = {
        "removed_exact_text_overlap": int(text_hit.sum()),
        "removed_domain_overlap": int(domain_hit.sum()),
    }
    return eval_df[keep].reset_index(drop=True), stats


def build_temporal_set(shift_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Rows with a source-attested event date strictly after the cutoff."""
    mask = shift_df["event_date"].notna() & (shift_df["event_date"] > cutoff)
    return shift_df[mask].reset_index(drop=True)


def finalize(df: pd.DataFrame, split_name: str = None) -> pd.DataFrame:
    out = df.copy()
    if split_name is not None:
        out["split"] = split_name
    out["label_name"] = out["label"].map(LABEL_NAMES)
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d").fillna("")
    return out[OUTPUT_COLUMNS]


def label_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()}


def source_label_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    tab = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    return {src: {LABEL_NAMES[int(l)]: int(n) for l, n in row.items()} for src, row in tab.iterrows()}


# ==============================================================================
# Orchestration
# ==============================================================================
def run(config_path: str = "params.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["base"]["random_seed"])
    set_seed(seed)

    task_cfg, raw_cfg, url_cfg = cfg["task"], cfg["raw_data"], cfg["url"]
    pvn_cfg, split_cfg = cfg["phishvn"], cfg["split"]
    out_dir = ensure_dir(cfg["output"]["processed_dir"])
    reports_dir = ensure_dir(cfg["output"]["reports_dir"])
    cutoff = pd.Timestamp(cfg["temporal"]["cutoff_date"])

    if task_cfg.get("include_sms", False):
        raise ValueError(
            "task.include_sms=true is not supported for the binary URL phishing task: "
            "SMSSpamCollection labels spam/ham, not phishing, and must not be mapped to phishing."
        )
    logger.info("SMSSpamCollection excluded from this task (task.include_sms=false); it stays DVC-versioned for provenance.")

    summary: Dict[str, Any] = {"task": task_cfg["name"], "url_representation": url_cfg["representation"],
                               "temporal_cutoff_date": str(cutoff.date())}

    # --- PhiUSIIL: training pool + held-out test
    phi = load_phiusiil(raw_cfg["phiusiil_path"])
    phi_stats: Dict[str, Any] = {"rows_raw": int(len(phi)), "labels_raw": label_counts(phi)}
    phi = build_url_columns(phi, url_cfg)
    phi, s = drop_invalid(phi, int(url_cfg["min_length"]))
    phi_stats.update(s)
    phi, s = dedupe_and_resolve_conflicts(phi)
    phi_stats.update(s)
    phi["split"] = assign_group_split(phi, split_cfg["train_ratio"], split_cfg["val_ratio"], split_cfg["test_ratio"], seed)
    phi_stats["rows_final"] = int(len(phi))
    phi_stats["has_path_rate_by_label"] = {LABEL_NAMES[k]: round(float(v), 4) for k, v in phi.groupby("label")["has_path"].mean().items()}
    summary["phiusiil"] = phi_stats

    splits = {}
    for name in ("train", "val", "test"):
        part = phi[phi["split"] == name]
        splits[name] = part
        summary[f"{name}_rows"] = int(len(part))
        summary[f"{name}_labels"] = label_counts(part)
        summary[f"{name}_domains"] = int(part["registrable_domain"].nunique())

    # --- PhishVN: evaluation only
    pvn, pvn_stats = load_phishvn(raw_cfg["phishvn_path"], pvn_cfg)
    pvn = build_url_columns(pvn, url_cfg)
    pvn, s = drop_invalid(pvn, int(url_cfg["min_length"]))
    pvn_stats.update(s)
    pvn, s = dedupe_and_resolve_conflicts(pvn)
    pvn_stats.update(s)
    pvn, s = remove_overlap(pvn, phi)
    pvn_stats.update(s)
    pvn_stats["rows_final"] = int(len(pvn))
    pvn_stats["labels_final"] = label_counts(pvn)
    pvn_stats["source_labels_final"] = source_label_counts(pvn)
    summary["phishvn"] = pvn_stats

    temporal = build_temporal_set(pvn, cutoff)
    summary["shift_test_rows"] = int(len(pvn))
    summary["shift_test_labels"] = label_counts(pvn)
    summary["temporal_test_rows"] = int(len(temporal))
    summary["temporal_test_labels"] = label_counts(temporal) if len(temporal) else {}
    summary["temporal_test_source_labels"] = source_label_counts(temporal) if len(temporal) else {}
    if len(temporal):
        summary["temporal_test_date_range"] = [str(temporal["event_date"].min().date()), str(temporal["event_date"].max().date())]
    pre = pvn[pvn["event_date"].notna() & (pvn["event_date"] <= cutoff)]
    summary["phishvn_dated_pre_cutoff_labels"] = label_counts(pre) if len(pre) else {}

    # --- Write outputs
    for name, part in splits.items():
        finalize(part).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    finalize(pvn, "shift_test").to_csv(os.path.join(out_dir, "shift_test.csv"), index=False)
    finalize(temporal, "temporal_test").to_csv(os.path.join(out_dir, "temporal_test.csv"), index=False)
    write_json(summary, os.path.join(reports_dir, "prepare_summary.json"))

    logger.info(
        "Prepared train=%d val=%d test=%d shift_test=%d temporal_test=%d",
        summary["train_rows"], summary["val_rows"], summary["test_rows"],
        summary["shift_test_rows"], summary["temporal_test_rows"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 prepare stage (binary URL phishing)")
    parser.add_argument("--config", default="params.yaml")
    run(parser.parse_args().config)
