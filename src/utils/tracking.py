"""
MLflow tracking helpers.

The tracking backend is chosen at runtime:
  1. the environment variable named by mlflow.tracking_uri_env (default MLFLOW_TRACKING_URI), else
  2. mlflow.default_tracking_uri from params.yaml (a local SQLite store for Phase 1).

A shared backend (e.g. DagsHub) can therefore be adopted later by exporting
MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD,
with no code changes. Credentials must never be written to params.yaml.
"""

import hashlib
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402


def resolve_tracking_uri(mlflow_cfg: Dict[str, Any]) -> str:
    env_name = mlflow_cfg.get("tracking_uri_env", "MLFLOW_TRACKING_URI")
    return os.environ.get(env_name) or mlflow_cfg.get("default_tracking_uri", "sqlite:///mlflow.db")


def configure_mlflow(mlflow_cfg: Dict[str, Any]) -> str:
    uri = resolve_tracking_uri(mlflow_cfg)
    mlflow.set_tracking_uri(uri)
    experiment_name = mlflow_cfg.get("experiment_name", "phase1-url-phishing")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_location = None
        if uri.startswith("sqlite:"):
            # Keep local artifacts next to the store rather than in an unpredictable CWD-relative path.
            artifact_location = os.path.abspath(mlflow_cfg.get("local_artifact_root", "mlartifacts"))
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    return uri


def file_md5(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def path_md5(path: str) -> str:
    """md5 of a file, or a combined md5 over all files (sorted) in a directory."""
    if os.path.isfile(path):
        return file_md5(path)
    h = hashlib.md5()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            p = os.path.join(root, name)
            h.update(os.path.relpath(p, path).encode())
            h.update(file_md5(p).encode())
    return h.hexdigest()


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Paths written by the pipeline itself. DVC deletes and regenerates these while stages run,
# so they are excluded when deciding whether the code/config differs from the commit.
PIPELINE_OUTPUT_PATHS = ("reports/phase1", "dvc.lock")


def lineage_tags(data_paths: Dict[str, str]) -> Dict[str, str]:
    """Git state, environment and input-data hashes, logged as run tags for lineage."""
    excludes = [f":(exclude){p}" for p in PIPELINE_OUTPUT_PATHS]
    tags = {
        "git.commit": _git("rev-parse", "HEAD") or "unknown",
        "git.branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # True if code/config/docs differ from the commit (pipeline outputs ignored)
        "git.dirty": str(bool(_git("status", "--porcelain", "--", ".", *excludes))),
        # True if anything differs, including outputs DVC is regenerating during the run
        "git.dirty_including_outputs": str(bool(_git("status", "--porcelain"))),
        "python.version": platform.python_version(),
        "platform": platform.platform(),
        "pipeline.phase": "phase1",
    }
    for name, path in data_paths.items():
        if path and os.path.exists(path):
            tags[f"data.{name}.path"] = path
            tags[f"data.{name}.md5"] = path_md5(path)
    if os.path.exists("dvc.lock"):
        tags["dvc.lock.md5"] = file_md5("dvc.lock")
    return tags


def package_versions(packages: Iterable[str]) -> Dict[str, str]:
    from importlib import metadata

    out = {}
    for p in packages:
        try:
            out[f"pkg.{p}"] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[f"pkg.{p}"] = "not-installed"
    out["pkg.python"] = sys.version.split()[0]
    return out


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts into dotted keys for mlflow.log_params."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(map(str, v))
        else:
            out[key] = v
    return out
