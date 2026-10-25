# Changelog

## Audit pass — 2026-10-24

- Added a project README with setup, data placement, execution commands, label-mode semantics, and known limitations.
- Added `requirements.txt` because the supplied archive did not contain one.
- Added `docs/requirements_traceability.md` with explicit status and evidence for the requested requirements.
- Changed default LDA topic selection to compare the broad grid but select the final model only from compliant candidates with 15–20 topics. Explicit `TSP_LDA_K` values outside this range now raise an error.
- Improved `scripts/verify_models.py` to fail with an actionable message when required saved model artifacts are absent instead of failing at a low-level missing-file read.
- Added `AUDIT_REPORT.md` and `VALIDATION_CHECKLIST.md`.
- Did not alter historical metric values or claim that missing model binaries were reverified.
