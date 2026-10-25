# Project Audit Report

**Audit date:** 2026-10-24  
**Scope:** Supplied `project.zip`; no raw source dataset or trained model binaries were supplied separately.

## Executive summary

The archive contains a substantial Python project, processed split CSVs, historical evaluation metrics, EDA figures, topic artifacts, scripts, and tests. It is **not fully submission-ready as supplied** because the source dataset and trained model binaries are missing; the saved DistilBERT verification record includes failed metric-match checks; the stored LDA artifact uses 12 topics rather than the required 15–20; the human topic review remains blank; and dashboard/browser behavior could not be fully exercised in the current environment.

I made focused packaging and safety-of-reporting improvements without changing historical metric values or inventing new model results:
- Added setup and usage documentation and a dependency manifest.
- Added requirements traceability and a validation checklist.
- Constrained future default LDA selection to the required 15–20 range.
- Made the saved-model verifier report missing model artifacts clearly.
- Preserved historical reports and metrics as historical artifacts.

## Archive inspection

Present:
- `app.py`, `src/`, `scripts/`, and `tests/`
- Four-class split CSVs (`train.csv`, `val.csv`, `test.csv`)
- Historical metrics, figures, leaderboard, and report
- Topic assignments, topic summary, review sheet, labels, and pyLDAvis HTML

Absent from the supplied ZIP:
- `models/four/` trained model binaries and label map
- Raw `twitter_training.csv` and `twitter_validation.csv`
- A dependency manifest and main setup README (added during this audit)

## Data and evaluation

The saved `data_validation.json` reports 74,682 training rows and 1,000 official validation rows, with four labels: Negative, Neutral, Positive, and Irrelevant. The processed split artifacts report 56,060 train, 9,894 internal validation, and 1,000 held-out test examples. This differs from the original story's 31,962-row, three-class specification. The code supports alternative three-class modes (`three_exclude` and `three_merge`), but a separate three-class experiment was not rerun here.

The historical evaluation protocol reserves the official validation file as the held-out test set and uses an internal split for model selection. This separation should be preserved. Historical classical results name Linear SVM as the selection winner based on validation macro-F1. The historical report's scores are not re-certified by this audit because the trained model files are absent.

The historical `verification.json` marks all listed classical metric checks true but marks DistilBERT's validation/test metric-match checks false. Since the model binaries are absent, I could not investigate the precision discrepancy or re-run inference. The stored results must not be presented as freshly verified by this audit.

## Topic modeling

The saved topic summary reports LDA with 12 topics, NMF with eight, and KMeans with ten clusters. The LDA count does not satisfy the stated 15–20 requirement. The source code now selects the best coherence only among compliant 15- and 20-topic candidates after computing the wider comparison grid; set `TSP_LDA_K` only to 15–20. Existing saved topic artifacts remain historical and must be regenerated from the raw data before claiming compliance.

The review sheet's `reviewer` and `notes` fields are blank. Human review is pending; no topic judgments were fabricated. The pyLDAvis HTML exists but was not rendered in a browser during this audit.

## Validation performed

- `python -m compileall -q src scripts app.py`: passed.
- `pytest -q tests/test_preprocessing.py tests/test_deep_models.py tests/test_topic_review.py tests/test_report_verification.py`: **14 passed**.
- Full `pytest -q`: collection stopped because `streamlit` is not installed (`ModuleNotFoundError` from `tests/test_app_smoke.py`). This is an environment/dependency limitation, not a successful full-suite run.
- Saved-model verification: cannot be run successfully because `models/four/` is absent. The verifier was updated to raise an actionable `FileNotFoundError` explaining this limitation rather than failing on an opaque label-map read.
- Streamlit dashboard workflows, live prediction, export workflows, and browser rendering: not verified in this environment.

## Submission-readiness decision

**Not yet fully submission-ready.** The source package is better documented and has focused tests passing, but the blockers listed in `VALIDATION_CHECKLIST.md` remain. At minimum, restore raw data and model binaries (or retrain from source data), install requirements, rerun the pipeline and verification scripts, regenerate the compliant topic artifacts, complete human review, and test the dashboard in the target environment. Do not claim a fully verified result until those steps pass.
