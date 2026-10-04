"""
Run the Phase 1 pipeline for an experiment variant without touching the main DVC outputs.

The main pipeline (dvc repro) remains the source of truth. A variant re-runs
prepare -> validate -> featurize -> train (both models) -> evaluate with parameter
overrides, writing to separate directories:

    data/processed/phase1/variants/<name>/   models/variants/<name>/   reports/phase1/variants/<name>/

The fully resolved configuration is saved as reports/phase1/variants/<name>/params_resolved.yaml,
and MLflow runs are tagged variant=<name> (run names prefixed "<name>:").

Examples:
    python scripts/run_variant.py --name host_only --set url.representation=host_only
    python scripts/run_variant.py --name xgb_lr0.3 --set models.xgboost.learning_rate=0.3
"""

import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import make_dataset, validate  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import evaluate, train  # noqa: E402
from src.utils.config import ensure_dir, load_config  # noqa: E402


def apply_override(cfg: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not key or not _:
        raise ValueError(f"Override must look like key.path=value, got '{assignment}'")
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in node:
            raise KeyError(f"Unknown parameter section '{part}' in '{key}'")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Unknown parameter '{key}'")
    node[parts[-1]] = yaml.safe_load(raw)


def build_config(name: str, overrides: list, base_path: str = "params.yaml") -> dict:
    cfg = copy.deepcopy(load_config(base_path))
    for assignment in overrides:
        apply_override(cfg, assignment)
    cfg["mlflow"]["variant"] = name
    processed = os.path.join("data", "processed", "phase1", "variants", name)
    cfg["output"] = {
        "processed_dir": processed,
        "features_dir": os.path.join(processed, "features"),
        "models_dir": os.path.join("models", "variants", name),
        "reports_dir": os.path.join("reports", "phase1", "variants", name),
    }
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Phase 1 experiment variant")
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = build_config(args.name, args.overrides)
    reports_dir = ensure_dir(cfg["output"]["reports_dir"])
    config_path = os.path.join(reports_dir, "params_resolved.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"# Resolved config for variant '{args.name}'. Overrides: {args.overrides}\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    make_dataset.run(config_path)
    if not validate.run(config_path):
        print(f"Validation failed for variant '{args.name}'; see {reports_dir}/validation.json")
        return 1
    build_features.run(config_path)
    for model_name in train.MODEL_NAMES:
        train.run(config_path, model_name)
    evaluate.run(config_path)
    print(f"Variant '{args.name}' done: {reports_dir}/model_comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
