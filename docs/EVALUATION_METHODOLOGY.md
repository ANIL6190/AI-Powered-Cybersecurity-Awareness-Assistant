# Evaluation methodology (Phase 1)

**Task:** binary URL classification, phishing (1) vs legitimate (0). Phishing is the positive class
for every metric.

**Models:** Model 1 (baseline) is TF-IDF (`char_wb`, 3–5-grams) → Logistic Regression. Model 2
(comparison) is the same TF-IDF matrices → XGBoost. The vectorizer is fitted on the PhiUSIIL training
split only.

**Threshold:** 0.5 for both models, fixed in advance and never tuned on a test set. Model selection
(the XGBoost tuning experiment) uses the validation split only.

## 1. Three different kinds of evaluation — do not mix them

| # | Evaluation | Data | What it measures | What it does **not** measure |
|---|---|---|---|---|
| 1 | **In-distribution held-out** | PhiUSIIL `test` (15% of registrable-domain groups, random assignment) | Generalisation to unseen domains drawn from the same collection as training | Any temporal effect; cross-source robustness |
| 1b | In-distribution, **no-path subset** | PhiUSIIL `test` rows whose URL has no path | The same, with the PhiUSIIL path shortcut removed (legitimate PhiUSIIL URLs never have paths) | Same as 1 |
| 2 | **Cross-source** | PhishVN verified core (`shift_test`), disjoint from PhiUSIIL by URL and registrable domain | Robustness to a separately collected dataset (Vietnamese-targeted URLs, different sources and labelling process) | Time on its own: source, region, URL format and label definition all change at once |
| 3 | **Time-stratified within PhishVN** | PhishVN NCSC phishing grouped by its source-attested detection date: on/before vs after the split date 2024-03-08 (`temporal_test`), and per year | Whether recall on PhishVN phishing changes across PhishVN's own time periods, with source and region held fixed | That the evaluation data is newer than the training data (see §2) |

The validation split (`val`, PhiUSIIL) is used only for XGBoost early stopping and model selection.

## 2. What the dates do and do not support

- **PhiUSIIL has no row-level dates.** Its file inside the Kaggle ZIP carries the timestamp 2024-03-08.
  This is a property of the file, not a collection date, so it is **not** treated as one. We therefore
  make **no claim** that any PhishVN row is newer than the training data, and no "train on the past,
  test on the future" claim.
- **PhishVN dates are source-attested**, but only for two sources:
  - `tinnhiemmang`: NCSC detection date of the phishing URL (2020-02-08 → 2025-02-18);
  - `tinnhiem_org`: NCSC certification date of a trusted organisation's site (2021-04-01 → 2026-03-25).

  `scraped_at` (when the dataset authors fetched the row) and ChongLuaDao's reconstructed dates are
  never used. Dates are parsed strictly and never imputed. Validation fails if a training row carries a
  date or a date comes from a non-attested source.
- **The split date 2024-03-08 is simply a split point inside PhishVN.** It divides the dated NCSC
  phishing roughly 85/15 (2,186 on/before, 370 after). It was originally chosen from the PhiUSIIL file
  timestamp, and that justification has been withdrawn. The per-year recall table is reported
  alongside so that conclusions do not depend on this single split point.
- **The legitimate class is not time-stratified**, except for the `tinnhiem_org` certification dates
  (an easy stratum, mostly `.gov.vn`). Precision, F1 and PR-AUC on `temporal_test` therefore describe
  "post-split NCSC phishing vs post-split certified organisations" only. **Phishing recall is the
  meaningful metric of evaluation 3.**

## 3. How to describe the results

Approved wording:

> "We report three evaluations. (1) In-distribution: a registrable-domain-grouped held-out split of
> PhiUSIIL. We also report the subset of URLs without a path, because PhiUSIIL's legitimate URLs never
> have paths. (2) Cross-source: the PhishVN verified core, a separately collected Vietnamese dataset that
> was never used for training. (3) Time-stratified: phishing recall on PhishVN's NCSC-detected phishing
> URLs, grouped by source-attested detection date (before/after 2024-03-08 and per year). PhiUSIIL has
> no row-level timestamps, so we do not claim that PhishVN data is newer than the training data; the
> time-stratified results show how cross-source recall varies across PhishVN detection periods."

Wording to avoid:
- "temporal hold-out", "train on past / test on future" or "newer than the training data";
- "temporal split" for the PhiUSIIL test set;
- "2024-03-08 is when PhiUSIIL was collected";
- "newer legitimate traffic";
- "PhishTank";
- "zero leakage guaranteed". Say instead that leakage is controlled by exact-text and registrable-domain
  disjointness checks.

## 4. Construction details

1. PhishVN is **evaluation-only**. Rows whose canonical URL (506) or registrable domain (525) occurs
   anywhere in PhiUSIIL are removed. The bronze tier (community labels, reconstructed dates) is excluded.
2. `shift_test` = all remaining verified PhishVN rows (2,556 phishing / 15,346 legitimate).
3. `temporal_test` = rows of `shift_test` with a source-attested date after the split date
   (370 NCSC phishing, 404 certified organisations).
4. Recall on the on/before-split phishing and per-year recall are computed from `shift_test`.

## 5. Metrics reported

For every model and set: phishing **recall**, **precision**, **F1**, **PR-AUC** (average precision),
ROC-AUC, accuracy, FPR, FNR and the confusion matrix. In addition:
- the held-out no-path subset;
- per-source recall and FPR on PhishVN;
- phishing recall before/after the split date and per detection year;
- the reference rule "URL has a path → phishing", which shows what the collection artefact alone achieves.

Outputs: `reports/phase1/metrics.json`, `reports/phase1/model_comparison.md`, `reports/phase1/plots/`,
the MLflow runs, and the variant reports under `reports/phase1/variants/`.

## 6. Candidate quality-gate metrics (recorded, not enforced)

`metrics.json → models.<model>.candidate_quality_gate_metrics`:
- `phishvn_post_split_phishing_recall` (risk R2: missed phishing on a different source and period);
- `heldout_phishing_f1` (risk R4).

Thresholds and automated promotion or blocking are Phase 2 and are **not** implemented.
