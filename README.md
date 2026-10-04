# AI-Powered Cybersecurity Awareness Assistant

An MLOps project that builds a **phishing URL classifier** with a reproducible, tracked and tested
pipeline. SDGs: 4 (Quality Education), 9 (Industry, Innovation and Infrastructure), 16 (Peace, Justice
and Strong Institutions).

**Current scope: Phase 1 — Development & Reproducibility.** Deployment (FastAPI, Docker, CI/CD, MLflow
Model Registry), monitoring (Evidently) and retraining are Phase 2/3 and are **not implemented yet**.

## Phase 1 at a glance

| Item | Implementation |
|---|---|
| Task | Binary URL classification: **phishing vs legitimate** |
| Model 1 (baseline) | TF-IDF (`char_wb`, 3–5-grams) → Logistic Regression |
| Model 2 (comparison) | Same TF-IDF → XGBoost |
| Training data | PhiUSIIL Phishing URL Dataset (registrable-domain-grouped 70/15/15 split) |
| Normal held-out evaluation | PhiUSIIL test split (random at domain level, **not temporal**) |
| In-distribution, no-path subset | PhiUSIIL test rows without a URL path (removes the PhiUSIIL path shortcut) |
| Cross-source evaluation | PhishVN v3.1.0 verified core (separate Vietnamese dataset, evaluation only) |
| Time-stratified evaluation | Phishing recall on PhishVN NCSC phishing by source-attested detection date (before/after 2024-03-08, per year). No claim that PhishVN is newer than the training data (PhiUSIIL is undated) |
| Data versioning | DVC (raw data + 7-stage pipeline; **no remote configured yet**) |
| Experiment tracking | MLflow (local SQLite store by default; any backend via `MLFLOW_TRACKING_URI`) |
| Data validation | 65 automated checks; failing checks stop the pipeline |
| Tests | PyTest, 45 tests on synthetic data |

Results: [`reports/phase1/model_comparison.md`](reports/phase1/model_comparison.md) (generated) and
[`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md).

## Repository layout

```
├── params.yaml                 # all pipeline parameters (single source of truth)
├── dvc.yaml / dvc.lock         # Phase 1 pipeline and its lock file
├── requirements.txt            # pinned dependencies (Python 3.13, see .python-version)
├── data/raw/*.dvc              # DVC pointers to the raw datasets (data itself is not in Git)
├── src/
│   ├── data/make_dataset.py    # prepare: load, canonicalise URLs, dedupe, grouped split, PhishVN evaluation sets
│   ├── data/validate.py        # validate: schema, labels, duplicates, leakage, date integrity
│   ├── eda/run_eda.py          # eda: figures + eda_summary.json
│   ├── features/build_features.py  # featurize: char TF-IDF fitted on train only
│   ├── models/train.py         # train_logreg / train_xgboost (+ MLflow run)
│   ├── models/evaluate.py      # evaluate: in-distribution, cross-source, time-stratified metrics, plots
│   ├── utils/                  # config, URL helpers, MLflow tracking helpers
│   └── data/preprocess.py      # legacy 3-class Stage 1 module (kept, not used by Phase 1)
├── notebooks/01_eda.ipynb      # EDA narrative (reads the eda stage outputs)
├── tests/                      # PyTest suite
├── docs/                       # datasets, evaluation methodology, risk register, results
├── reports/phase1/             # generated metrics, validation report, plots, EDA figures
└── archive/                    # legacy Stage 1 artefacts kept for traceability
```

## Setup

```bash
python3 -m venv .venv                 # Python 3.13
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_nltk.py          # only needed for the legacy module's tests
```

## Data

Raw data is versioned with DVC but **no DVC remote is configured yet**, so `dvc pull` is not available.
Obtain the files listed in [`docs/DATASETS.md`](docs/DATASETS.md) and place them as:

```
data/raw/PhiUSIIL_Phishing_URL_Dataset.csv      # md5 f3c27559e55c6c104e4d83857f0e7608
data/raw/SMSSpamCollection                      # md5 1949b64a224790d01335c2bf8a0e48b2 (provenance only)
data/raw/phishvn_v3.1.0/                        # from PhishVN_v3.1.0_open.zip (see docs/DATASETS.md)
```

Then run `dvc status data/raw/*.dvc` to confirm that the files match the versioned hashes.

## Run the pipeline

```bash
source .venv/bin/activate
dvc repro            # prepare → validate → eda → featurize → train_logreg / train_xgboost → evaluate
dvc metrics show     # headline metrics
dvc plots show       # PR curves and confusion matrices (HTML)
dvc dag              # pipeline graph
```

Each stage can also be run on its own, e.g. `python -m src.models.train --model logreg`.

Experiment variants (outputs kept separate from the main pipeline, MLflow runs tagged `variant=<name>`):

```bash
python scripts/run_variant.py --name host_only --set url.representation=host_only
```

## Experiment tracking (MLflow)

By default, runs go to a local SQLite store (`mlflow.db`, artifacts in `mlartifacts/`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Each model run logs:
- params: TF-IDF, model, URL handling, split, PhishVN split date, seed;
- validation, held-out (+ no-path subset), cross-source and time-stratified metrics;
- lineage tags: git commit and dirty flag, md5 of raw and processed data, `dvc.lock` md5, package versions;
- artifacts: params, prepare/validation/feature summaries, plots, comparison table;
- the fitted TF-IDF + classifier pipeline.

To use a shared server later (e.g. DagsHub), export `MLFLOW_TRACKING_URI` (and
`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`). No code change is needed, and credentials must
never be put in `params.yaml`.

## Tests

```bash
pytest tests/ -q
```

## Documentation

- [`docs/DATASETS.md`](docs/DATASETS.md): sources, licences, checksums, approval record, limitations
- [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md): the three kinds of evaluation,
  what the dates do and do not support, and approved wording
- [`docs/RISK_REGISTER.md`](docs/RISK_REGISTER.md): R1–R4, separating what Phase 1 implements from what is
  planned
- [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md): results summary
- [`DATA_CLEANING_REPORT.md`](DATA_CLEANING_REPORT.md): legacy Stage 1 report (superseded, with corrections)

## Attribution

PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024). PhishVN v3.1.0 (Thai Nguyen Vu, UTC2;
CC BY 4.0; DOI 10.17632/b97hxbxtpd.4), which builds on NCSC "Tin Nhiem Mang" and the Tranco list
(Le Pochat et al., NDSS 2019). SMS Spam Collection (Almeida & Gómez Hidalgo, UCI).
