# Datasets — acquisition, provenance and approval (Phase 1)

This file records where every dataset came from, exactly which bytes we use, what each dataset
is used for, and its known limitations. All raw files live in `data/raw/` and are versioned with
DVC (`data/raw/*.dvc`); they are never committed to Git. **No DVC remote is configured yet**, so
the raw bytes currently exist only in the local DVC cache (see "Open items").

| Dataset | Role in Phase 1 | Rows used | Licence |
|---|---|---|---|
| PhiUSIIL Phishing URL Dataset | Training pool + normal held-out test | 231,746 after cleaning | CC BY 4.0 (UCI listing; confirm) |
| PhishVN v3.1.0 (open tier) | **Evaluation only**: cross-source and time-stratified (within PhishVN) evaluation | 18,997 verified rows in, 17,902 after overlap removal | CC BY 4.0 |
| SMSSpamCollection v.1 | **Not used by the model.** Versioned for provenance only | 0 | CC BY 4.0 (UCI) |

---

## 1. PhiUSIIL Phishing URL Dataset

- **Origin:** Prasad, A. & Chandra, S., *PhiUSIIL: A diverse security profile empowered phishing URL
  detection framework based on similarity index and incremental learning*, Computers & Security 136 (2024),
  also listed in the UCI Machine Learning Repository. Obtained as a Kaggle download (`archive.zip`).
  *(Confirm the exact Kaggle page / UCI ID and licence text with the team before submission.)*
- **Local source file:** `~/Downloads/archive.zip` (SHA-256 `b848046a…d2ad0f7`), member
  `PhiUSIIL_Phishing_URL_Dataset.csv`, ZIP entry timestamp **2024-03-08 16:41:52**.
- **Copied to:** `data/raw/PhiUSIIL_Phishing_URL_Dataset.csv` (extracted, not modified).
- **md5:** `f3c27559e55c6c104e4d83857f0e7608`, 54,160,642 bytes, 235,795 rows + header, 55 columns.
- **Labels:** `label` 1 = legitimate (134,850), 0 = phishing (100,945). The pipeline inverts this to
  1 = phishing.
- **Columns used:** only `URL` and `label`. The other 53 pre-computed columns (e.g. `URLSimilarityIndex`,
  page-content features) are deliberately ignored: several are known near-perfect proxies of the label
  and the required models are TF-IDF-only.

### Difference from the legacy Stage 1 file

The legacy `dvc.lock` (preserved in `archive/legacy_stage1_dvc.lock`) recorded a PhiUSIIL file with
md5 `769ba530b3629c4c2f9896abf642aefa` and 56,854,345 bytes. Our file has **the same row count
(235,795) and the same header**, but different bytes (md5 `f3c27559…`, 2,693,703 bytes smaller). The most
likely explanation is that the earlier copy was re-saved or re-encoded, but this cannot be confirmed
without that file. We use the original Kaggle download because its provenance is clearer. The
regenerated `dvc.lock` records the file we actually use.

### Known limitations (measured in EDA)

- **Collection-format shortcut.** In the raw file, 100% of legitimate URLs are `https://www.<domain>`
  with no path and no trailing slash, while phishing URLs are 51% `http://`, 59% without `www.`,
  36% with a trailing slash. The pipeline removes scheme, leading `www.` and trailing slash for every
  dataset (`src/utils/urls.py`).
- **Residual path shortcut.** Even after canonicalisation, 0% of legitimate URLs but ~28% of phishing URLs
  have a path. A path therefore implies phishing in this dataset. This is an artefact of how the
  legitimate class was collected (home pages only), and `reports/phase1/model_comparison.md` reports
  what the rule "has a path → phishing" achieves on its own.
- **No row-level timestamps.** PhiUSIIL cannot be split temporally. The ZIP entry timestamp
  (2024-03-08) is a property of the file, not a collection date, and is **not** used to make any
  claim about when PhiUSIIL was collected or about PhishVN being newer than it.
- **Hosting platforms.** Thousands of phishing URLs sit on shared hosting (`*.web.app`,
  `*.firebaseapp.com`, `*.repl.co`, `*.workers.dev`). Grouping uses the Public Suffix List's private
  section so that each tenant is its own group.

---

## 2. PhishVN v3.1.0 — Vietnamese URL phishing dataset (open tier)

**PhishVN is not PhishTank.** It replaces the originally planned PhishTank `verified_online.csv`,
which was not available. Unlike PhishTank, it is an academic dataset with documented sources,
confidence tiers and source-attested event dates.

- **Citation:** Thai Nguyen Vu (University of Transport and Communications, Ho Chi Minh City Campus),
  *PhishVN: A Time-Stamped Vietnamese URL Phishing Dataset with Impersonation-Scenario Labels and
  Confidence Tiers*, v3.1.0, DOI `10.17632/b97hxbxtpd.4`, https://github.com/vuthainguyen1602/phishvn.
- **Licence:** CC BY 4.0 (data). Attribution is required to PhishVN, to NCSC "Tin Nhiem Mang" and to the
  Tranco list (Le Pochat et al., NDSS 2019). The licence notes must be read before use. The gated
  HTML/screenshot tier was **not** obtained or used.
- **Local source file:** `~/Downloads/PhishVN_v3.1.0_open.zip` (SHA-256 `308351e9…25d0ab5`).
- **Copied to `data/raw/phishvn_v3.1.0/`:** `dataset_url.csv`, `abuse_type.csv`, `datasheet.md`,
  `schema.md`, `data_sources.md`, `LICENSE`, `CITATION.cff`, `README.md`, `MANIFEST.txt`.
  The official splits and the paraphrase-attack files were not copied because they are not used.
- **Integrity:** SHA-256 of `dataset_url.csv` (`8e1c6325…6eade`) and `abuse_type.csv`
  (`3e354d64…389a`) match the publisher's `MANIFEST.txt`.

### Composition (`dataset_url.csv`, 53,116 rows, 29 columns)

| Source | Label | Tier | Rows | `collected_at` meaning | Used? |
|---|---|---|---:|---|---|
| `tinnhiemmang` (NCSC blacklist) | phishing | gold 1,688 / silver 899 | 2,587 | Detection date attested by NCSC, `DD/MM/YYYY`, 2020-02-08 → 2025-02-18 | Yes, dated |
| `tinnhiem_org` (NCSC trusted-organisation registry) | benign | gold | 2,026 | Certification date attested by NCSC, 2021-04-01 → 2026-03-25 | Yes, dated |
| `tinnhiem_web` (NCSC registry) | benign | gold | 5,879 | empty | Yes, undated |
| `tranco` (global top-list sample) | benign | silver | 5,993 | empty | Yes, undated |
| `tranco_vn` (`.vn` top-list slice) | benign | silver | 2,512 | empty | Yes, undated |
| `chongluadao` (community blocklist) | phishing | bronze | 33,823 | ISO date *reconstructed* from web archives for ~50% | **No** |
| `openphish` | phishing | bronze | 296 | empty | **No** |

- **Timestamp field used:** `collected_at`, and only for sources whose date is attested by the source
  itself (`phishvn.attested_date_sources` = `tinnhiemmang`, `tinnhiem_org`). It is parsed strictly as
  `DD/MM/YYYY`; values that fail to parse stay empty (0 such values in v3.1.0).
- **Not used as a timestamp:** `scraped_at` (when the PhishVN authors fetched the row) and the
  reconstructed ChongLuaDao dates.
- **Bronze tier excluded** (`phishvn.tiers: [gold, silver]`). It is community-labelled, the PhishVN audit
  judged ~12% of positive-arm rows legitimate (mostly bronze), and its dates are reconstructed.
- **Overlap removal:** PhishVN rows whose canonical URL (506) or registrable domain (525) also appears in
  the PhiUSIIL pool are removed from the evaluation sets.

### Known limitations (from the PhishVN datasheet and our EDA)

- **The positive label means "listed by a Vietnamese anti-fraud feed".** Credential phishing is about 42.5% of
  defensible positives in the publisher's audit; the rest is gambling, investment, counterfeit-shop and
  similar fraud. This is truer of bronze than of the NCSC rows we use, but recall on PhishVN should be read
  as *fraud-URL* recall.
- **URL format differs from PhiUSIIL, and also between PhishVN classes.** PhishVN rows are mostly bare
  hostnames (no path). In the raw file, 46% of phishing rows start with `www.` versus 0.5% of
  legitimate rows, and 12% of legitimate rows (the `tinnhiem_org` source) carry `https://` versus 0.8% of
  phishing rows. Canonicalisation removes both differences. The missing paths are a genuine distribution
  shift relative to PhiUSIIL.
- **`.vn` shortcut.** In our `shift_test`, P(legitimate | `.vn` TLD) = 0.99. The publisher added Tranco
  samples to counter this, but it remains a feature of the data.
- **The NCSC feed has been dormant since 2025-02-18.** The newest dated phishing is 2025-02-18.
- **Legitimate class mostly undated.** Only `tinnhiem_org` (certified organisations, mostly `.gov.vn`,
  an easy stratum) carries dates.

---

## 3. SMSSpamCollection v.1

- **Origin:** Almeida, T.A. & Gómez Hidalgo, J.M., UCI Machine Learning Repository (SMS Spam Collection).
- **Local source file:** `~/Downloads/sms+spam+collection.zip` (SHA-256 `1587ea43…963ce3`), member
  `SMSSpamCollection`.
- **Copied to:** `data/raw/SMSSpamCollection`; md5 `1949b64a224790d01335c2bf8a0e48b2`, 477,907 bytes.
  This is **identical** to the file recorded in the legacy `dvc.lock`.
- **Content:** 5,574 English SMS messages (4,827 ham, 747 spam), collected around 2011–2012.
- **Phase 1 use:** **none.** Its labels are spam vs ham, not phishing, and it is a different input type
  (SMS text, not URLs). `task.include_sms: false` excludes it, and setting the flag to `true` raises an
  error instead of mapping spam to phishing. It stays DVC-versioned so the legacy Stage 1 work remains
  traceable.

---

## 4. Approval record

| Item | Status |
|---|---|
| PhiUSIIL use approved by faculty | ☐ pending |
| PhishVN use approved by faculty (replaces PhishTank) | ☐ pending |
| SMSSpamCollection retained for provenance only | ☐ pending |
| Licences reviewed (CC BY 4.0 attribution in README/report) | ☐ pending |
| Actual download dates confirmed (files in `~/Downloads` show 2026-10-04 timestamps) | ☐ pending |

## 5. Open items

- **DVC remote not configured** (provider decision pending). Until it is, a clean clone cannot
  `dvc pull`; the raw files must be obtained from the sources above and checked against the md5s in
  `data/raw/*.dvc`. Storage needed: ~67 MB of raw data plus ~65 MB of pipeline outputs per version.
- Confirm PhiUSIIL's licence and listing details, and its collection period from the paper. Only a
  documented collection period could support a train-before / test-after claim; none is made now.
