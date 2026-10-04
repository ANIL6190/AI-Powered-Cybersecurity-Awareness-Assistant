"""
Phase 1 `evaluate` stage.

Three distinct kinds of evaluation (see docs/EVALUATION_METHODOLOGY.md):

  1. In-distribution  — `test`: PhiUSIIL held-out split (registrable-domain grouped, random; NOT temporal).
  2. Cross-source     — `shift_test`: PhishVN verified core, a separately collected Vietnamese dataset.
  3. Time-stratified, within PhishVN only — recall on PhishVN NCSC phishing grouped by its
     source-attested detection date: before vs after temporal.cutoff_date (a split point inside
     PhishVN) and per year. Because the models never see PhishVN and PhiUSIIL has no row-level dates,
     this measures how cross-source recall changes across PhishVN time periods. It does NOT show
     that the evaluation data is newer than the training data.
     `temporal_test` = PhishVN rows dated after the split point (NCSC phishing + NCSC-certified
     organisations, the only dated legitimate stratum).

Writes reports/phase1/metrics.json (DVC metrics), reports/phase1/plots/*.png (DVC plots),
reports/phase1/model_comparison.md, and logs everything to each model's MLflow run.
"""

import argparse
import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.models.metrics import (  # noqa: F401  (compute_* re-exported for callers/tests)
    breakdown_by_source,
    compute_metrics,
    compute_recall_only,
    has_path_rule,
    recall_by_period,
)
from src.utils.config import ensure_dir, get_logger, load_config, read_json, write_json

logger = get_logger("evaluate")

MODEL_NAMES = ("logreg", "xgboost")
EVAL_SETS = ("test", "shift_test", "temporal_test")
SET_LABELS = {
    "test": "PhiUSIIL held-out (in-distribution)",
    "shift_test": "PhishVN verified core (cross-source)",
    "temporal_test": "PhishVN dated rows after split date (cross-source, time-stratified)",
}


# ==============================================================================
# Plots
# ==============================================================================
def plot_pr_curves(results: Dict[str, Dict[str, Any]], plots_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for eval_set in EVAL_SETS:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for model_name, by_set in results.items():
            y, proba = by_set[eval_set]["y"], by_set[eval_set]["proba"]
            if len(np.unique(y)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y, proba)
            ap = average_precision_score(y, proba)
            ax.plot(rec, prec, label=f"{model_name} (PR-AUC={ap:.3f})")
        base = float(np.mean(next(iter(results.values()))[eval_set]["y"]))
        ax.axhline(base, ls="--", lw=1, color="grey", label=f"prevalence={base:.3f}")
        ax.set_xlabel("Recall (phishing)")
        ax.set_ylabel("Precision (phishing)")
        ax.set_title(f"Precision-recall — {eval_set}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        path = os.path.join(plots_dir, f"pr_curve_{eval_set}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_confusion_matrices(metrics: Dict[str, Dict[str, Any]], plots_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for model_name, by_set in metrics.items():
        fig, axes = plt.subplots(1, len(EVAL_SETS), figsize=(4 * len(EVAL_SETS), 3.6))
        for ax, eval_set in zip(axes, EVAL_SETS):
            m = by_set[eval_set]
            cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=10)
            ax.set_xticks([0, 1], ["legitimate", "phishing"])
            ax.set_yticks([0, 1], ["legitimate", "phishing"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(eval_set)
        fig.suptitle(f"Confusion matrices — {model_name}")
        fig.tight_layout()
        path = os.path.join(plots_dir, f"confusion_matrix_{model_name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_recall_by_year(metrics: Dict[str, Any], plots_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name in MODEL_NAMES:
        by_year = metrics["models"][model_name]["phishvn_phishing_recall_by_year"]
        years = list(by_year)
        ax.plot(years, [by_year[y]["recall"] for y in years], marker="o", label=model_name)
    counts = metrics["models"][MODEL_NAMES[0]]["phishvn_phishing_recall_by_year"]
    ax.set_xticks(range(len(counts)), [f"{y}\n(n={v['n_phishing']})" for y, v in counts.items()], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Phishing recall")
    ax.set_title("PhishVN NCSC phishing — recall by detection year\n(cross-source; models trained on PhiUSIIL only)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(plots_dir, "phishvn_recall_by_year.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]


# ==============================================================================
# Report
# ==============================================================================
def comparison_markdown(metrics: Dict[str, Any], cutoff: str, variant: str = "main") -> str:
    lines = [
        "# Phase 1 model comparison" + ("" if variant == "main" else f" — variant `{variant}`"),
        "",
        "Generated by `src/models/evaluate.py`. Positive class = phishing. Threshold = "
        f"{metrics['threshold']}. URL representation: `{metrics.get('url_representation', 'canonical_url')}`.",
        "",
        "## 1. In-distribution and cross-source evaluation",
        "",
        "| Model | Evaluation set | n | Phishing recall | Phishing precision | Phishing F1 | PR-AUC | ROC-AUC | FPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        for key in EVAL_SETS:
            m = metrics["models"][model_name][key]
            lines.append(
                f"| {model_name} | {SET_LABELS[key]} | {m['n']:,} | {m['recall']:.4f} | {m['precision']:.4f} | "
                f"{m['f1']:.4f} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['false_positive_rate']:.4f} |"
            )
    lines += [
        "",
        "PhiUSIIL held-out rows **without a URL path** (the path shortcut cannot help; identical input for "
        "canonical_url and host_only):",
        "",
        "| Model | n | Phishing recall | Phishing precision | Phishing F1 | PR-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        m = metrics["models"][model_name]["test_no_path_subset"]
        lines.append(f"| {model_name} | {m['n']:,} | {m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | "
                     f"{m['pr_auc']:.4f} | {m['false_positive_rate']:.4f} |")
    lines += [
        "",
        f"## 2. Time-stratified recall within PhishVN NCSC phishing (split date {cutoff})",
        "",
        "Same source and region, different detection periods. The models never saw PhishVN, so these are "
        "cross-source results stratified by time. They do not show that PhishVN is newer than the training data.",
        "",
        "| Model | Recall, detected <= split date | Recall, detected > split date |",
        "|---|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        m = metrics["models"][model_name]
        lines.append(f"| {model_name} | {m['phishvn_phishing_pre_cutoff']['recall']:.4f} "
                     f"(n={m['phishvn_phishing_pre_cutoff']['n_phishing']}) | "
                     f"{m['temporal_phishing_only']['recall']:.4f} (n={m['temporal_phishing_only']['n_phishing']}) |")
    years = list(metrics["models"][MODEL_NAMES[0]]["phishvn_phishing_recall_by_year"])
    lines += ["", "| Model | " + " | ".join(years) + " |", "|---|" + "---:|" * len(years)]
    for model_name in MODEL_NAMES:
        by_year = metrics["models"][model_name]["phishvn_phishing_recall_by_year"]
        lines.append(f"| {model_name} | " + " | ".join(f"{by_year[y]['recall']:.3f}" for y in years) + " |")
    counts = metrics["models"][MODEL_NAMES[0]]["phishvn_phishing_recall_by_year"]
    lines.append("| n | " + " | ".join(str(counts[y]["n_phishing"]) for y in years) + " |")
    lines += [
        "",
        "## 3. Reference: format shortcut ('has a path -> phishing')",
        "",
        "| Evaluation set | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for key, ref in metrics["reference_has_path_rule"].items():
        lines.append(f"| {key} | {ref['precision']:.4f} | {ref['recall']:.4f} | {ref['f1']:.4f} |")
    lines += [
        "",
        "Notes: the PhiUSIIL split is random at the registrable-domain level and is not a temporal evaluation. "
        "PhiUSIIL has no row-level dates, so no claim is made about PhishVN being newer than the training data. "
        "The legitimate rows of temporal_test are only NCSC-certified organisations (an easy stratum); the other "
        "legitimate PhishVN sources are undated.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: str = "params.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    out = cfg["output"]
    threshold = float(cfg["evaluation"]["threshold"])
    cutoff = pd.Timestamp(cfg["temporal"]["cutoff_date"])
    reports_dir = out["reports_dir"]
    plots_dir = ensure_dir(os.path.join(reports_dir, "plots"))
    attested = set(cfg["phishvn"]["attested_date_sources"])

    frames = {s: pd.read_csv(os.path.join(out["processed_dir"], f"{s}.csv"), keep_default_na=False,
                             dtype={"event_date": str}) for s in EVAL_SETS}
    X = {s: sp.load_npz(os.path.join(out["features_dir"], f"X_{s}.npz")).tocsr() for s in EVAL_SETS}

    shift = frames["shift_test"]
    shift_dates = pd.to_datetime(shift["event_date"], format="%Y-%m-%d", errors="coerce")  # "" -> NaT
    dated_phish = ((shift["label"] == 1) & shift_dates.notna() & shift["source"].isin(attested)).to_numpy()
    pre_mask = dated_phish & (shift_dates <= cutoff).to_numpy()
    shift_years = shift_dates.dt.year.where(pd.Series(dated_phish)).astype("Int64")
    temporal_phish_mask = (frames["temporal_test"]["label"] == 1).to_numpy()

    all_metrics: Dict[str, Any] = {
        "threshold": threshold,
        "temporal_split_date": str(cutoff.date()),
        "url_representation": cfg["url"]["representation"],
        "models": {},
    }
    curves: Dict[str, Dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        model = joblib.load(os.path.join(out["models_dir"], f"{model_name}.joblib"))
        per_set: Dict[str, Any] = {}
        curves[model_name] = {}
        for s in EVAL_SETS:
            proba = model.predict_proba(X[s])[:, 1]
            y = frames[s]["label"].to_numpy()
            per_set[s] = compute_metrics(y, proba, threshold)
            curves[model_name][s] = {"y": y, "proba": proba}
            if s == "test":
                # Rows where the URL has no path: the 'path -> phishing' shortcut cannot help here,
                # and canonical_url and host_only give the same input string for these rows.
                no_path = (frames[s]["has_path"] == 0).to_numpy()
                per_set["test_no_path_subset"] = compute_metrics(y[no_path], proba[no_path], threshold)
            if s == "shift_test":
                per_set["shift_test_by_source"] = breakdown_by_source(frames[s], proba, threshold)
                per_set["phishvn_phishing_pre_cutoff"] = compute_recall_only(y[pre_mask], proba[pre_mask], threshold)
                per_set["phishvn_phishing_recall_by_year"] = recall_by_period(y, proba, shift_years, threshold)
            if s == "temporal_test":
                per_set["temporal_phishing_only"] = compute_recall_only(
                    y[temporal_phish_mask], proba[temporal_phish_mask], threshold)
                per_set["temporal_test_by_source"] = breakdown_by_source(frames[s], proba, threshold)
        train_report = read_json(os.path.join(reports_dir, f"train_{model_name}.json"))
        per_set["val"] = train_report["validation"]
        per_set["candidate_quality_gate_metrics"] = {
            # Recorded only. Thresholds and enforcement belong to Phase 2.
            "phishvn_post_split_phishing_recall": per_set["temporal_phishing_only"]["recall"],
            "heldout_phishing_f1": per_set["test"]["f1"],
        }
        all_metrics["models"][model_name] = per_set
        logger.info("%s: test F1=%.4f | PhishVN recall=%.4f | PhishVN post-split phishing recall=%.4f", model_name,
                    per_set["test"]["f1"], per_set["shift_test"]["recall"], per_set["temporal_phishing_only"]["recall"])

    all_metrics["reference_has_path_rule"] = {s: has_path_rule(frames[s]) for s in EVAL_SETS}

    write_json(all_metrics, os.path.join(reports_dir, "metrics.json"))
    plot_paths = (plot_pr_curves(curves, plots_dir) + plot_confusion_matrices(all_metrics["models"], plots_dir)
                  + plot_recall_by_year(all_metrics, plots_dir))
    comparison_path = os.path.join(reports_dir, "model_comparison.md")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison_markdown(all_metrics, str(cutoff.date()), cfg["mlflow"].get("variant", "main")))

    log_to_mlflow(cfg, all_metrics, plot_paths, comparison_path)
    return all_metrics


def _flat_numeric(d: Dict[str, Any], prefix: str) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
            out[f"{prefix}_{k}"] = float(v)
    return out


def log_to_mlflow(cfg: Dict[str, Any], all_metrics: Dict[str, Any], plot_paths: List[str], comparison_path: str) -> None:
    import mlflow
    from src.utils import tracking

    tracking.configure_mlflow(cfg["mlflow"])
    reports_dir = cfg["output"]["reports_dir"]
    for model_name in MODEL_NAMES:
        run_id = read_json(os.path.join(reports_dir, f"train_{model_name}.json")).get("mlflow_run_id")
        if not run_id:
            logger.warning("No MLflow run id for %s; skipping MLflow logging.", model_name)
            continue
        m = all_metrics["models"][model_name]
        with mlflow.start_run(run_id=run_id):
            metrics = {}
            metrics.update(_flat_numeric(m["test"], "test"))
            metrics.update(_flat_numeric(m["test_no_path_subset"], "test_no_path"))
            metrics.update(_flat_numeric(m["shift_test"], "phishvn"))
            metrics.update(_flat_numeric(m["temporal_test"], "phishvn_post_split"))
            metrics.update(_flat_numeric(m["temporal_phishing_only"], "phishvn_post_split_phish"))
            metrics.update(_flat_numeric(m["phishvn_phishing_pre_cutoff"], "phishvn_pre_split_phish"))
            for year, r in m["phishvn_phishing_recall_by_year"].items():
                metrics[f"phishvn_phish_recall_{year}"] = r["recall"]
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(os.path.join(reports_dir, "metrics.json"), artifact_path="evaluation")
            mlflow.log_artifact(comparison_path, artifact_path="evaluation")
            for p in plot_paths:
                mlflow.log_artifact(p, artifact_path="evaluation/plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 evaluation stage")
    parser.add_argument("--config", default="params.yaml")
    run(parser.parse_args().config)
