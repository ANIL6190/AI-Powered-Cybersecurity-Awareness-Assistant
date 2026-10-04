"""
Phase 1 `validate` stage.

Runs data-quality and leakage checks on the prepared datasets and writes
reports/phase1/validation.json. Exits with a non-zero status if any check fails,
which stops `dvc repro` before features or models are built (risk R3).
"""

import argparse
import os
import sys
from typing import Any, Callable, Dict, List

import pandas as pd

from src.utils.config import get_logger, load_config, write_json

logger = get_logger("validate")

REQUIRED_COLUMNS = ["model_text", "label", "label_name", "dataset", "source", "tier",
                    "registrable_domain", "has_path", "event_date", "url_raw", "split"]
SPLIT_FILES = ["train", "val", "test", "shift_test", "temporal_test"]


def load_splits(processed_dir: str) -> Dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(os.path.join(processed_dir, f"{name}.csv"), dtype={"event_date": str},
                          keep_default_na=False)
        for name in SPLIT_FILES
    }


class Checker:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def check(self, name: str, passed: bool, detail: Any = None) -> None:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        (logger.info if passed else logger.error)("%s %s %s", "PASS" if passed else "FAIL", name,
                                                  "" if detail is None else detail)

    @property
    def ok(self) -> bool:
        return all(r["passed"] for r in self.results)


def _overlap(a: pd.Series, b: pd.Series) -> int:
    return len(set(a) & set(b))


def run_checks(splits: Dict[str, pd.DataFrame], cfg: Dict[str, Any]) -> Checker:
    c = Checker()
    cutoff = pd.Timestamp(cfg["temporal"]["cutoff_date"])
    vcfg = cfg["validation"]

    # --- Schema and content
    for name, df in splits.items():
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        c.check(f"{name}.schema", not missing, {"missing_columns": missing})
        if missing:
            continue
        c.check(f"{name}.non_empty", len(df) > 0 or name == "temporal_test", {"rows": len(df)})
        c.check(f"{name}.labels_binary", set(df["label"].unique()) <= {0, 1},
                {"labels": sorted(map(int, df["label"].unique()))})
        name_ok = ((df["label"] == 1) == (df["label_name"] == "phishing")).all()
        c.check(f"{name}.label_name_consistent", bool(name_ok))
        empty_text = int((df["model_text"].astype(str).str.strip() == "").sum())
        c.check(f"{name}.no_empty_model_text", empty_text == 0, {"empty": empty_text})
        empty_group = int((df["registrable_domain"].astype(str).str.strip() == "").sum())
        c.check(f"{name}.no_empty_group_key", empty_group == 0, {"empty": empty_group})
        dups = int(df["model_text"].duplicated().sum())
        c.check(f"{name}.no_duplicate_model_text", dups == 0, {"duplicates": dups})
        conflicts = int((df.groupby("model_text")["label"].nunique() > 1).sum())
        c.check(f"{name}.no_label_conflicts", conflicts == 0, {"conflicting_texts": conflicts})
        c.check(f"{name}.both_classes_present", df["label"].nunique() == 2 or len(df) == 0,
                {"classes": sorted(map(int, df["label"].unique()))})

    # --- Task definition
    all_rows = pd.concat(splits.values(), ignore_index=True)
    sms_rows = int(all_rows["source"].astype(str).str.contains("SMS", case=False).sum())
    c.check("task.sms_excluded", (not cfg["task"].get("include_sms", False)) and sms_rows == 0, {"sms_rows": sms_rows})
    pool_datasets = set(pd.concat([splits[s] for s in ("train", "val", "test")])["dataset"])
    c.check("task.training_pool_is_phiusiil_only", pool_datasets == {"phiusiil"}, {"datasets": sorted(pool_datasets)})
    eval_datasets = set(pd.concat([splits["shift_test"], splits["temporal_test"]])["dataset"])
    c.check("task.eval_sets_are_phishvn_only", eval_datasets <= {"phishvn"}, {"datasets": sorted(eval_datasets)})

    # --- Leakage: exact text and registrable domain
    train = splits["train"]
    for other in ("val", "test", "shift_test", "temporal_test"):
        c.check(f"leakage.text.train_vs_{other}", _overlap(train["model_text"], splits[other]["model_text"]) == 0,
                {"overlap": _overlap(train["model_text"], splits[other]["model_text"])})
        c.check(f"leakage.domain.train_vs_{other}",
                _overlap(train["registrable_domain"], splits[other]["registrable_domain"]) == 0,
                {"overlap": _overlap(train["registrable_domain"], splits[other]["registrable_domain"])})
    c.check("leakage.domain.val_vs_test", _overlap(splits["val"]["registrable_domain"], splits["test"]["registrable_domain"]) == 0,
            {"overlap": _overlap(splits["val"]["registrable_domain"], splits["test"]["registrable_domain"])})
    pool = pd.concat([splits[s] for s in ("train", "val", "test")])
    c.check("leakage.domain.pool_vs_shift", _overlap(pool["registrable_domain"], splits["shift_test"]["registrable_domain"]) == 0,
            {"overlap": _overlap(pool["registrable_domain"], splits["shift_test"]["registrable_domain"])})

    # --- Class balance across the random-but-grouped split
    rate = {s: float(splits[s]["label"].mean()) for s in ("train", "val", "test")}
    drift = max(abs(rate["val"] - rate["train"]), abs(rate["test"] - rate["train"]))
    c.check("split.class_ratio_drift", drift <= vcfg["max_class_ratio_drift"],
            {"phishing_rate": {k: round(v, 4) for k, v in rate.items()}, "max_drift": round(drift, 4),
             "allowed": vcfg["max_class_ratio_drift"]})

    # --- Temporal integrity (no fabricated or pre-cutoff dates)
    temporal = splits["temporal_test"]
    tdates = pd.to_datetime(temporal["event_date"], format="%Y-%m-%d", errors="coerce")
    c.check("temporal.min_rows", len(temporal) >= vcfg["min_temporal_rows"], {"rows": len(temporal),
            "required": vcfg["min_temporal_rows"]})
    c.check("temporal.all_rows_dated", bool(tdates.notna().all()), {"undated": int(tdates.isna().sum())})
    c.check("temporal.all_after_cutoff", bool((tdates > cutoff).all()),
            {"cutoff": str(cutoff.date()), "min_date": str(tdates.min().date()) if len(tdates) else None})
    attested = set(cfg["phishvn"]["attested_date_sources"])
    c.check("temporal.only_attested_sources", set(temporal["source"]) <= attested, {"sources": sorted(set(temporal["source"]))})
    phi_dated = int((pool["event_date"].astype(str) != "").sum())
    c.check("temporal.phiusiil_has_no_dates", phi_dated == 0, {"dated_rows": phi_dated})
    shift_dates = splits["shift_test"]
    unattested_dated = int(((shift_dates["event_date"].astype(str) != "") & ~shift_dates["source"].isin(attested)).sum())
    c.check("temporal.no_dates_from_unattested_sources", unattested_dated == 0, {"rows": unattested_dated})
    return c


def run(config_path: str = "params.yaml") -> bool:
    cfg = load_config(config_path)
    splits = load_splits(cfg["output"]["processed_dir"])
    checker = run_checks(splits, cfg)
    n_fail = sum(not r["passed"] for r in checker.results)
    report = {
        "status": "PASSED" if checker.ok else "FAILED",
        "checks_total": len(checker.results),
        "checks_failed": n_fail,
        "rows": {k: int(len(v)) for k, v in splits.items()},
        "checks": checker.results,
    }
    write_json(report, os.path.join(cfg["output"]["reports_dir"], "validation.json"))
    logger.info("Validation %s (%d checks, %d failed)", report["status"], len(checker.results), n_fail)
    return checker.ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 data validation stage")
    parser.add_argument("--config", default="params.yaml")
    sys.exit(0 if run(parser.parse_args().config) else 1)
