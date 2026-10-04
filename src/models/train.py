"""
Phase 1 `train_logreg` / `train_xgboost` stages.

  Model 1 (baseline):   TF-IDF (char_wb 3-5) -> Logistic Regression
  Model 2 (comparison): TF-IDF (char_wb 3-5) -> XGBoost

Both models use exactly the same TF-IDF matrices. Class weighting is derived from
the training split only; the validation split is used for XGBoost early stopping
and for reporting, never the test or shift sets.

Each run is tracked in MLflow (params, validation metrics, lineage tags, artifacts
and — if mlflow.log_models — the fitted TF-IDF + classifier pipeline). The run id is
written to reports/phase1/train_<model>.json so the evaluate stage can log to the same run.
"""

import argparse
import os
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

from src.features.build_features import to_text_array
from src.models.metrics import compute_metrics
from src.utils.config import ensure_dir, get_logger, load_config, set_seed, write_json
from src.utils import tracking

logger = get_logger("train")

MODEL_NAMES = ("logreg", "xgboost")


def build_inference_pipeline(vectorizer, model) -> Pipeline:
    """TF-IDF + classifier pipeline that accepts a DataFrame/Series/list of URLs (used for MLflow logging)."""
    return Pipeline([
        ("to_text", FunctionTransformer(to_text_array, validate=False)),
        ("tfidf", vectorizer),
        ("clf", model),
    ])


def load_features(features_dir: str, split: str):
    X = sp.load_npz(os.path.join(features_dir, f"X_{split}.npz")).tocsr()
    y = np.load(os.path.join(features_dir, f"y_{split}.npy")).astype(int)
    return X, y


def build_model(name: str, model_cfg: Dict[str, Any], y_train: np.ndarray, seed: int):
    if name == "logreg":
        return LogisticRegression(
            C=model_cfg["C"],
            solver=model_cfg["solver"],
            max_iter=model_cfg["max_iter"],
            class_weight=model_cfg["class_weight"],
            random_state=seed,
        )
    if name == "xgboost":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if model_cfg.get("balance_classes", True) and n_pos else 1.0
        return XGBClassifier(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            learning_rate=model_cfg["learning_rate"],
            subsample=model_cfg["subsample"],
            colsample_bytree=model_cfg["colsample_bytree"],
            min_child_weight=model_cfg["min_child_weight"],
            reg_lambda=model_cfg["reg_lambda"],
            tree_method=model_cfg["tree_method"],
            max_bin=model_cfg.get("max_bin", 256),
            early_stopping_rounds=model_cfg["early_stopping_rounds"],
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=model_cfg["n_jobs"],
            random_state=seed,
        )
    raise ValueError(f"Unknown model '{name}', expected one of {MODEL_NAMES}")


def fit_xgboost(model: XGBClassifier, X_train, y_train, X_val, y_val) -> XGBClassifier:
    """
    Train with the native API and early stopping on the validation split, then return an
    XGBClassifier wrapping the booster (so predict_proba / sklearn pipelines work unchanged).

    The sklearn wrapper builds the eval set as a QuantileDMatrix referencing the training
    matrix; with 100k sparse TF-IDF columns that made each evaluation round ~35x slower than
    training. A plain DMatrix for validation gives identical metrics at negligible cost.
    """
    import xgboost as xgb

    params = model.get_xgb_params()
    num_rounds = params.pop("n_estimators", None) or model.n_estimators
    early_stopping = params.pop("early_stopping_rounds", None) or model.early_stopping_rounds
    params = {k: v for k, v in params.items() if v is not None}
    params.setdefault("objective", "binary:logistic")
    dtrain = xgb.QuantileDMatrix(X_train, y_train, max_bin=params.get("max_bin", 256))
    dval = xgb.DMatrix(X_val, y_val)
    booster = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=[(dval, "val")],
                        early_stopping_rounds=early_stopping, verbose_eval=False)
    fitted = XGBClassifier(**model.get_params())
    fitted.load_model(bytearray(booster.save_raw("json")))
    return fitted


def fit_model(name: str, model, X_train, y_train, X_val, y_val):
    if name == "xgboost":
        return fit_xgboost(model, X_train, y_train, X_val, y_val)
    model.fit(X_train, y_train)
    return model


def run(config_path: str, model_name: str) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg["base"]["random_seed"])
    set_seed(seed)
    out = cfg["output"]
    models_dir = ensure_dir(out["models_dir"])
    model_cfg = cfg["models"][model_name]
    threshold = float(cfg["evaluation"]["threshold"])

    X_train, y_train = load_features(out["features_dir"], "train")
    X_val, y_val = load_features(out["features_dir"], "val")

    model = build_model(model_name, model_cfg, y_train, seed)
    logger.info("Training %s on %d rows x %d TF-IDF features...", model_name, X_train.shape[0], X_train.shape[1])
    start = time.time()
    model = fit_model(model_name, model, X_train, y_train, X_val, y_val)
    train_seconds = round(time.time() - start, 2)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, val_proba, threshold)
    logger.info("%s validation: recall=%.4f f1=%.4f pr_auc=%.4f (%.1fs)", model_name,
                val_metrics["recall"], val_metrics["f1"], val_metrics["pr_auc"], train_seconds)

    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    extra: Dict[str, Any] = {"train_seconds": train_seconds, "n_train": int(X_train.shape[0]),
                             "n_features": int(X_train.shape[1])}
    if model_name == "xgboost":
        extra["best_iteration"] = int(getattr(model, "best_iteration", model_cfg["n_estimators"] - 1))
        extra["scale_pos_weight"] = float(model.get_params()["scale_pos_weight"])

    run_id = log_to_mlflow(cfg, model_name, model, val_metrics, extra)

    report = {"model": model_name, "mlflow_run_id": run_id, "threshold": threshold,
              "validation": val_metrics, **extra}
    write_json(report, os.path.join(out["reports_dir"], f"train_{model_name}.json"))
    return report


def log_to_mlflow(cfg: Dict[str, Any], model_name: str, model, val_metrics: Dict[str, Any],
                  extra: Dict[str, Any]) -> str:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    out = cfg["output"]
    uri = tracking.configure_mlflow(cfg["mlflow"])
    logger.info("MLflow tracking URI: %s", uri)

    data_paths = {
        "phiusiil_raw": cfg["raw_data"]["phiusiil_path"],
        "phishvn_raw": cfg["raw_data"]["phishvn_path"],
        "train": os.path.join(out["processed_dir"], "train.csv"),
        "val": os.path.join(out["processed_dir"], "val.csv"),
        "test": os.path.join(out["processed_dir"], "test.csv"),
        "shift_test": os.path.join(out["processed_dir"], "shift_test.csv"),
        "temporal_test": os.path.join(out["processed_dir"], "temporal_test.csv"),
    }

    variant = cfg["mlflow"].get("variant", "main")
    run_name = f"{model_name}-tfidf-char" if variant == "main" else f"{variant}:{model_name}-tfidf-char"
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.set_tags({
            **tracking.lineage_tags(data_paths),
            **tracking.package_versions(["scikit-learn", "xgboost", "mlflow", "pandas", "numpy", "dvc"]),
            "model.family": model_name,
            "model.role": "baseline" if model_name == "logreg" else "comparison",
            "variant": variant,
            "task": cfg["task"]["name"],
            "features": "tfidf_char_wb",
        })
        params = {
            "model": model_name,
            "seed": cfg["base"]["random_seed"],
            "threshold": cfg["evaluation"]["threshold"],
            "temporal.cutoff_date": cfg["temporal"]["cutoff_date"],
            **tracking.flatten(cfg["models"][model_name], "model"),
            **tracking.flatten(cfg["features"]["tfidf"], "tfidf"),
            **tracking.flatten(cfg["url"], "url"),
            **tracking.flatten(cfg["split"], "split"),
            "phishvn.tiers": ",".join(cfg["phishvn"]["tiers"]),
            "task.include_sms": cfg["task"]["include_sms"],
        }
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({k: float(v) for k, v in extra.items() if isinstance(v, (int, float))})

        for path in ("params.yaml",
                     os.path.join(out["reports_dir"], "prepare_summary.json"),
                     os.path.join(out["reports_dir"], "validation.json"),
                     os.path.join(out["reports_dir"], "features_summary.json")):
            if os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="lineage")

        if cfg["mlflow"].get("log_models", True):
            vectorizer = joblib.load(os.path.join(out["models_dir"], "tfidf_vectorizer.joblib"))
            pipeline = build_inference_pipeline(vectorizer, model)
            example = pd.DataFrame({"url": ["example.com/login/verify", "wikipedia.org"]})
            signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=example,
                code_paths=["src"],
                # skops is impractically slow/large for a 100k-entry vocabulary; pickle-based artifacts
                # must only be loaded from a trusted (our own) tracking store.
                serialization_format=cfg["mlflow"].get("model_serialization", "cloudpickle"),
            )
        return active.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 model training stage")
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    args = parser.parse_args()
    run(args.config, args.model)
