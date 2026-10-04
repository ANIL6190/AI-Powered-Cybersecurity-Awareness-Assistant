"""
Metric helpers shared by the train and evaluate stages (phishing = positive class).

Kept separate from evaluate.py so that changes to the evaluation report do not
invalidate the DVC training stages.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Binary metrics with phishing (1) as the positive class."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    n_pos, n_neg = int(tp + fn), int(tn + fp)
    return {
        "n": int(len(y_true)),
        "n_phishing": n_pos,
        "n_legitimate": n_neg,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else float("nan"),
        "false_positive_rate": float(fp / n_neg) if n_neg else float("nan"),
        "false_negative_rate": float(fn / n_pos) if n_pos else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compute_recall_only(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """For phishing-only slices: recall / miss rate are the only meaningful metrics."""
    y_true = np.asarray(y_true).astype(int)
    if not (y_true == 1).all():
        raise ValueError("compute_recall_only expects phishing-only labels")
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(y_pred.sum())
    n = int(len(y_true))
    return {"n_phishing": n, "tp": tp, "fn": n - tp,
            "recall": float(tp / n) if n else float("nan"),
            "false_negative_rate": float((n - tp) / n) if n else float("nan")}


def breakdown_by_source(df: pd.DataFrame, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Recall per phishing source and false-positive rate per legitimate source."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    out: Dict[str, Any] = {}
    for src, idx in df.groupby("source").groups.items():
        pos = df.index.get_indexer(idx)
        y = df["label"].to_numpy()[pos]
        p = pred[pos]
        if (y == 1).all():
            out[src] = {"class": "phishing", "n": int(len(y)), "recall": float(p.mean())}
        elif (y == 0).all():
            out[src] = {"class": "legitimate", "n": int(len(y)), "false_positive_rate": float(p.mean())}
    return out


def recall_by_period(labels: np.ndarray, proba: np.ndarray, periods: pd.Series, threshold: float) -> Dict[str, Any]:
    """Phishing recall for each period label (e.g. year); rows with a missing period are skipped."""
    out: Dict[str, Any] = {}
    labels = np.asarray(labels)
    proba = np.asarray(proba)
    periods = pd.Series(periods).reset_index(drop=True)
    for period in sorted(periods.dropna().unique()):
        mask = (periods == period).fillna(False).to_numpy(dtype=bool) & (labels == 1)
        if mask.any():
            out[str(period)] = compute_recall_only(labels[mask], proba[mask], threshold)
    return out


def has_path_rule(df: pd.DataFrame) -> Dict[str, Any]:
    """Reference shortcut: 'URL has a path -> phishing'. Shows how much a format artefact alone achieves."""
    proba = df["has_path"].to_numpy(dtype=float)
    m = compute_metrics(df["label"].to_numpy(), proba, 0.5)
    return {k: m[k] for k in ("precision", "recall", "f1", "false_positive_rate")}
