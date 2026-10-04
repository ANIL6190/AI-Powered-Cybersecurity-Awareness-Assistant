# Archive — legacy Stage 1 artefacts (kept for traceability)

Nothing here is used by the Phase 1 pipeline. These files are kept unchanged so the earlier work can
still be inspected and compared. They can be removed once the team has reviewed and approved the new
pipeline.

| File | What it is | Why it was archived |
|---|---|---|
| `clean_datasets.py` | Early standalone cleaning script (writes `cleaned_data/`, labels ham=0/spam=1) | Second, conflicting pipeline outside DVC. Its PhishTank timestamp parsing is no longer applicable (PhishTank was replaced by PhishVN), and timestamp handling now lives in `src/data/make_dataset.py`. |
| `legacy_stage1_params.yaml` | `params.yaml` of the 3-class Stage 1 pipeline | Replaced by the Phase 1 `params.yaml`. `src/data/preprocess.py` can still be run with `--config archive/legacy_stage1_params.yaml` if the legacy raw files are placed at the paths it names. |
| `legacy_stage1_dvc.yaml` | The single-stage `preprocess` DVC pipeline | Replaced by the Phase 1 multi-stage `dvc.yaml` |
| `legacy_stage1_dvc.lock` | Lock file of that pipeline | Records the md5s of the legacy raw files, including the PhiUSIIL copy whose bytes differ from ours (see `docs/DATASETS.md`) and PhishTank `verified_online.csv`, which we do not have |

The legacy module `src/data/preprocess.py` itself stays in place, because its tests still run and its
helpers may be reused. Its sequence tokenizer (deep learning) code is untouched. The only change is a
minimal bug fix so `<url>`/`<phone>`/… placeholders survive NLTK tokenization.

**Known defect of the legacy pipeline (do not reuse its outputs):** URL rows were passed through SMS text
cleaning, so every URL became the single token `"url"` and TF-IDF carried no URL information. The Phase 1
pipeline does not route URLs through that code path.
