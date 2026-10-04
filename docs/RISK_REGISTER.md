# Risk register — AI-Powered Cybersecurity Awareness Assistant

Risk score = Probability × Impact (1–5 scales). Levels: 15–25 Severe/Major as assigned by the team.

**Status key:** **Phase 1 — implemented** means it exists in this repository and is exercised by
`dvc repro` / `pytest`. **Planned (Phase 2/3)** means it is not implemented yet and must not be
claimed as existing.

| ID | Risk | P | I | Score | Level |
|---|---|---:|---:|---:|---|
| R1 | Model degradation due to phishing drift | 4 | 5 | 20 | Severe |
| R2 | False negatives on unseen phishing patterns | 4 | 5 | 20 | Severe |
| R3 | Data leakage / misleading evaluation | 3 | 5 | 15 | Major |
| R4 | Unsuitable model deployed | 3 | 4 | 12 | Major |

---

## R1 — Model degradation due to phishing drift

| Mitigation | Status |
|---|---|
| Time-stratified recall within PhishVN: NCSC phishing grouped by source-attested detection date (before/after 2024-03-08 and per year), with source and region held fixed | **Phase 1 — implemented** |
| Note: PhiUSIIL is undated, so Phase 1 cannot show performance on data newer than the training data | Limitation |
| Cross-source shift evaluation (`shift_test`) with per-source recall/FPR | **Phase 1 — implemented** |
| EDA timeline of attested dates; documented split-date rationale and its limits | **Phase 1 — implemented** |
| Evidently data/prediction drift monitoring on simulated production traffic | Planned (Phase 3) |
| Logging of production predictions | Planned (Phase 2/3) |
| Retraining trigger and feedback loop | Planned (Phase 3) |

**Residual risk:** the post-split set is small (370 phishing URLs), Vietnamese-targeted, and from one
feed that has been dormant since 2025-02-18. Recall varies across PhishVN detection years, which is
consistent with drift sensitivity but is not a train-past/test-future measurement and not a monitoring
system.

## R2 — False negatives on unseen phishing patterns

| Mitigation | Status |
|---|---|
| Phishing recall, F1, PR-AUC, FNR and confusion matrices on in-distribution (incl. no-path subset), cross-source and time-stratified sets | **Phase 1 — implemented** |
| Recall on unseen-domain phishing: all evaluation sets are registrable-domain-disjoint from training | **Phase 1 — implemented** |
| `phishvn_post_split_phishing_recall` recorded as a candidate quality-gate metric | **Phase 1 — recorded only** |
| Threshold selection policy (e.g. recall floor at a fixed FPR) | Planned (Phase 2) |
| Enforced recall gate in CI | Planned (Phase 2) |

**Residual risk:** the model only sees the URL string. Phishing on reputable hosts (e.g.
`sites.google.com`) and bare look-alike domains remain hard. PhiUSIIL's legitimate URLs are home pages
only, so URL paths are an artefact signal (see the reference rule in `model_comparison.md`).

## R3 — Data leakage / misleading evaluation

| Mitigation | Status |
|---|---|
| DVC versioning of all raw data (`data/raw/*.dvc`) and pipeline outputs (`dvc.lock`) | **Phase 1 — implemented** (no remote yet) |
| URL canonicalisation applied identically to all datasets (removes the scheme / `www.` / trailing-slash shortcut) | **Phase 1 — implemented** |
| Deduplication on the model input, and dropping of label-conflicting texts | **Phase 1 — implemented** |
| Registrable-domain-grouped train/val/test split (PSL private suffixes so hosting tenants are separate groups) | **Phase 1 — implemented** |
| PhishVN evaluation rows overlapping PhiUSIIL by text or domain removed | **Phase 1 — implemented** |
| TF-IDF fitted on train only; class weights from train only; test sets never used for tuning | **Phase 1 — implemented** |
| Validation stage (65 checks) that fails `dvc repro` on schema, label, duplicate, leakage or date-integrity violations | **Phase 1 — implemented** |
| No fabricated timestamps; strict date parsing; validation of date provenance | **Phase 1 — implemented** |
| MLflow lineage: git commit/dirty flag, md5 of raw and processed data, `dvc.lock` md5, params, package versions | **Phase 1 — implemented** |
| Shortcut reporting ("has path → phishing" reference rule; format-artefact EDA; no-path held-out subset; host-only variant) | **Phase 1 — implemented** |
| Legacy URL-collapse bug (every URL became the token "url") fixed and regression-tested | **Phase 1 — implemented** |
| DVC remote so others can reproduce from a clean clone | Pending decision |

**Residual risk:** dataset-construction artefacts in PhiUSIIL (home-page-only legitimate URLs) inflate
in-distribution scores by about 7 F1 points (held-out F1 0.847 vs 0.779 on rows without a path). Held-out numbers
must be reported together with the no-path subset and the cross-source and time-stratified numbers.

## R4 — Unsuitable model deployed

| Mitigation | Status |
|---|---|
| PyTest suite (45 tests: data, validation, features, training, evaluation, legacy) | **Phase 1 — implemented** |
| Baseline vs comparison model tracked in MLflow with identical features and data lineage | **Phase 1 — implemented** |
| Candidate gate metrics recorded (`phishvn_post_split_phishing_recall`, `heldout_phishing_f1`) | **Phase 1 — recorded only** |
| MLflow Model Registry and promotion workflow | Planned (Phase 2) |
| Automated quality gate blocking promotion in CI/CD (GitHub Actions) | Planned (Phase 2) |
| FastAPI service + Docker image | Planned (Phase 2) |
| Model card and governance sign-off | Planned (Phase 3) |

## Additional risks identified during Phase 1

| ID | Risk | Note |
|---|---|---|
| R5 | Label-definition mismatch | PhishVN "phishing" = listed by an anti-fraud feed (only ~42.5% credential phishing in the publisher's audit). Recall on PhishVN is fraud-URL recall. |
| R6 | Pickle-based model artefacts | MLflow models are logged with cloudpickle (skops was impractical for a 100k-term vocabulary). Only load models from our own tracking store. |
| R7 | Reproducibility from a clean clone | No DVC remote yet; the raw data must be re-obtained and checked against the `.dvc` md5s. |
