# Phase 1 results — initial model comparison

**Run:** local `dvc repro` plus `scripts/run_variant.py` variants on 2026-10-04 (Python 3.13.9, macOS).
Numbers come from `reports/phase1/metrics.json` and `reports/phase1/model_comparison.md` (main pipeline)
and `reports/phase1/variants/*/` (experiments). If this file disagrees with those generated files, the
generated files are correct. The MLflow runs (experiment `phase1-url-phishing`) carry `git.dirty=True`
because the work has not been committed; re-run after committing to get clean lineage tags.

Positive class = phishing. Threshold 0.5, fixed in advance. Model selection used the validation split
only. The three kinds of evaluation are defined in `docs/EVALUATION_METHODOLOGY.md` and are kept
separate below.

## 1. Data

| Set | Rows | Phishing | Legitimate | Kind of evaluation |
|---|---:|---:|---:|---|
| train / val / test | 164,113 / 34,157 / 33,476 | 69,419 / 13,835 / 13,685 | 94,694 / 20,322 / 19,791 | PhiUSIIL, registrable-domain-grouped random split |
| test, no-path subset | 29,528 | 9,737 | 19,791 | In-distribution with the path shortcut removed |
| shift_test | 17,902 | 2,556 | 15,346 | Cross-source (PhishVN verified core, disjoint by URL and domain) |
| temporal_test | 774 | 370 | 404 | PhishVN rows dated after the split date 2024-03-08 (time-stratified, within PhishVN) |

Validation: **PASSED** (65/65 checks). Features: TF-IDF `char_wb` 3–5-grams, 100,000 terms, fitted on
train only.

## 2. Main results (canonical URL representation)

**XGBoost configuration:** selected by validation PR-AUC from a small experiment (§4): learning rate 0.3,
max depth 10, early stopping on val (best round 446).

### 2.1 In-distribution (PhiUSIIL held-out)

| Model | Set | Recall | Precision | F1 | PR-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|
| **LogReg (baseline)** | full test | 0.787 | 0.916 | **0.847** | **0.933** | 0.050 |
| | no-path subset | 0.703 | 0.873 | **0.779** | **0.881** | 0.050 |
| **XGBoost (comparison)** | full test | 0.759 | 0.928 | 0.835 | 0.927 | 0.040 |
| | no-path subset | 0.667 | 0.890 | 0.762 | 0.871 | 0.040 |

The no-path subset is the more honest in-distribution figure. In PhiUSIIL a URL path only ever
appears on phishing URLs, and the rule "has a path → phishing" alone reaches precision 1.00 at recall 0.29.

### 2.2 Cross-source (PhishVN verified core)

| Model | Recall | Precision | F1 | PR-AUC | FPR | FPR on Tranco top sites | FPR on `.vn` top sites |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogReg | **0.737** | 0.405 | **0.523** | **0.540** | 0.180 | 0.271 | 0.200 |
| XGBoost | 0.651 | 0.392 | 0.489 | 0.507 | 0.168 | 0.229 | 0.178 |

Performance drops sharply on a different source, mainly through false positives on popular legitimate
sites.

### 2.3 Time-stratified recall within PhishVN NCSC phishing (cross-source)

The models never saw PhishVN, and PhiUSIIL has no row-level dates. These figures show how
cross-source recall varies across PhishVN detection periods. **They do not show that PhishVN is
newer than the training data.**

| Model | On/before 2024-03-08 (n=2,186) | After 2024-03-08 (n=370) |
|---|---:|---:|
| LogReg | 0.757 | 0.622 |
| XGBoost | 0.669 | 0.541 |

| Detection year | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|---:|---:|
| n | 389 | 1,115 | 555 | 124 | 334 | 39 |
| LogReg recall | 0.784 | 0.723 | 0.822 | 0.685 | 0.632 | 0.564 |
| XGBoost recall | 0.740 | 0.620 | 0.726 | 0.629 | 0.545 | 0.538 |

Recall is lower for 2023–2025 detections than for 2020–2022 for both models. This is consistent with
phishing patterns drifting (R1), but it is not a controlled measurement: the mix of campaigns and
impersonated sectors also changes by year, and the 2023 and 2025 groups are small.

Candidate quality-gate metrics (recorded only): LogReg `phishvn_post_split_phishing_recall` 0.622,
`heldout_phishing_f1` 0.847; XGBoost 0.541 / 0.835.

## 3. Host-only URL experiment (path removed)

Variant `host_only` (`url.representation=host_only`): the model sees only the hostname (without
`www.`). The registrable-domain split assignment is identical to the main run.

| Model | Representation | Full test F1 / PR-AUC | **No-path subset F1 / PR-AUC** | PhishVN recall / PR-AUC / FPR | PhishVN post-split recall |
|---|---|---:|---:|---:|---:|
| LogReg | canonical URL | 0.847 / 0.933 | **0.779 / 0.881** | 0.737 / 0.540 / 0.180 | 0.622 |
| LogReg | host only | 0.793 / 0.895 | **0.778 / 0.880** | 0.774 / 0.486 / 0.287 | 0.692 |
| XGBoost | canonical URL | 0.835 / 0.927 | **0.762 / 0.871** | 0.651 / 0.507 / 0.168 | 0.541 |
| XGBoost | host only | 0.777 / 0.887 | **0.766 / 0.871** | 0.681 / 0.475 / 0.254 | 0.603 |

**Findings**
1. **The path shortcut inflates the headline held-out score by about 5–7 F1 points.** Full-test F1 0.847
   falls to 0.779 on rows without a path (LogReg). This inflation comes from how PhiUSIIL was
   collected, not from model skill.
2. **On comparable rows, removing the path changes nothing.** On the no-path subset (identical input
   strings for both representations), the two representations give practically the same F1 and PR-AUC
   for both models. The canonical model is therefore not relying on the path for URLs that have none.
3. **Host-only is not better cross-source.** Its higher PhishVN recall comes from a shifted
   operating point (FPR 0.18 → 0.29 for LogReg), and its threshold-free PR-AUC on PhishVN is *lower*
   (0.540 → 0.486). The path carries some genuine signal as well as the artefact.
4. **Decision:** keep `canonical_url` as the primary representation. Report the no-path subset next to
   the full held-out score, and keep `host_only` as a documented sensitivity analysis.

Caveat: host-only collapses many phishing URLs that share a host, so its full test set is smaller
(11,945 vs 13,685 phishing) and its full-test numbers are not directly comparable. The no-path subset is
the like-for-like comparison (29,439 vs 29,528 rows; the small difference comes from deduplication).

## 4. Small XGBoost tuning experiment

Three settings, each one run, selected by **validation PR-AUC only** (test and PhishVN numbers shown
for transparency, not used for selection):

| Setting | Best round | Val PR-AUC | Test F1 | Test PR-AUC | PhishVN recall | PhishVN PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| lr 0.1, depth 6 (original, preliminary) | 599 (cap) | 0.9258 | 0.819 | 0.921 | 0.540 | 0.512 |
| lr 0.3, depth 6 | 579 | 0.9309 | 0.833 | 0.928 | 0.636 | 0.518 |
| lr 0.3, depth 6, colsample 0.4 | 594 | 0.9316 | 0.834 | 0.927 | 0.646 | 0.516 |
| **lr 0.3, depth 10 (selected)** | 446 | **0.9317** | 0.836 | 0.928 | 0.651 | 0.507 |
| *LogReg baseline, for reference* | — | *0.9323* | *0.847* | *0.933* | *0.737* | *0.540* |

Tuning improved XGBoost clearly over the preliminary run. The three lr-0.3 settings are within noise
of each other. **Tuned XGBoost still does not beat the Logistic Regression baseline** on any headline
metric. This was a deliberately small search (three settings, one seed), not an exhaustive one.

## 5. Conclusions

- **Baseline:** TF-IDF + Logistic Regression is the better Phase 1 model. It has the best held-out F1 and
  PR-AUC, the best cross-source recall and PR-AUC, and the best recall on the most recent PhishVN detections.
- **Comparison:** TF-IDF + XGBoost (lightly tuned) has slightly higher precision and lower FPR, but
  misses more phishing. Under R2 (false negatives are the severe risk), it is the weaker option.
- **Generalisation is the main weakness, not in-distribution accuracy.** Cross-source precision is about 0.40,
  and recall on recent PhishVN phishing is about 0.62 even for the better model.

## 6. Limitations

- One run per configuration, one seed, no confidence intervals.
- PhiUSIIL collection artefacts: home-page-only legitimate URLs (path shortcut), quantified above.
- No train-past/test-future evaluation is possible, because PhiUSIIL is undated.
- PhishVN "phishing" = listed by an anti-fraud feed, so its recall is fraud-URL recall. The time-stratified
  groups are small for 2023 (124) and 2025 (39).
- No DVC remote yet.
