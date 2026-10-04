"""
Phase 1 `featurize` stage: character-level TF-IDF on the canonical URL.

The vectorizer is fitted on the TRAIN split only and then applied unchanged to
val, test, shift_test and temporal_test, so no vocabulary or IDF statistics leak
from evaluation data.
"""

import argparse
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.config import ensure_dir, get_logger, load_config, write_json

logger = get_logger("build_features")

EVAL_SPLITS = ["train", "val", "test", "shift_test", "temporal_test"]


def to_text_array(X: Any) -> np.ndarray:
    """Accept a DataFrame (first column), Series, list or array of URLs and return a 1-D string array.
    Used as the first step of the logged inference pipeline so it accepts tabular input."""
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return np.asarray(X, dtype=str).ravel()


def make_vectorizer(tfidf_cfg: Dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=tfidf_cfg["analyzer"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        max_features=tfidf_cfg["max_features"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        lowercase=tfidf_cfg["lowercase"],
        dtype=np.float32,
    )


def load_split(processed_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(processed_dir, f"{name}.csv"), keep_default_na=False)


def run(config_path: str = "params.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    processed_dir = cfg["output"]["processed_dir"]
    features_dir = ensure_dir(cfg["output"]["features_dir"])
    models_dir = ensure_dir(cfg["output"]["models_dir"])

    frames = {name: load_split(processed_dir, name) for name in EVAL_SPLITS}
    vectorizer = make_vectorizer(cfg["features"]["tfidf"])
    logger.info("Fitting TF-IDF (%s, ngram_range=%s) on %d training URLs only...",
                cfg["features"]["tfidf"]["analyzer"], cfg["features"]["tfidf"]["ngram_range"], len(frames["train"]))
    vectorizer.fit(frames["train"]["model_text"].astype(str))

    summary: Dict[str, Any] = {"vocabulary_size": int(len(vectorizer.vocabulary_)), "fitted_on": "train", "splits": {}}
    for name, df in frames.items():
        X = vectorizer.transform(df["model_text"].astype(str)).tocsr()
        sp.save_npz(os.path.join(features_dir, f"X_{name}.npz"), X)
        np.save(os.path.join(features_dir, f"y_{name}.npy"), df["label"].to_numpy(dtype=np.int8))
        summary["splits"][name] = {
            "rows": int(X.shape[0]),
            "nnz": int(X.nnz),
            "empty_rows": int((X.getnnz(axis=1) == 0).sum()),
        }
        logger.info("%s: shape=%s nnz=%d", name, X.shape, X.nnz)

    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    write_json(summary, os.path.join(cfg["output"]["reports_dir"], "features_summary.json"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 TF-IDF feature stage")
    parser.add_argument("--config", default="params.yaml")
    run(parser.parse_args().config)
