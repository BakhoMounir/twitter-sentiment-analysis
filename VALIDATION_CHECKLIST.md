# Final Validation Checklist

- [x] Source files compile with the current Python interpreter.
- [x] Focused unit tests pass: preprocessing, deep-model helpers, topic review, report verification.
- [x] Historical dataset counts and labels inspected in saved metadata.
- [x] Historical DistilBERT verification record inspected; its metric-match checks are false.
- [x] Human topic review status treated as pending.
- [x] Missing raw data and trained model binaries documented.
- [x] Missing-model verifier error made actionable.
- [x] Default LDA selector constrained to 15–20 topics for a future rerun.
- [ ] Full pytest suite including Streamlit smoke test (blocked: Streamlit not installed in audit environment).
- [ ] Reproduce classical saved-model metrics (blocked: model binaries absent).
- [ ] Reproduce DistilBERT metrics and resolve precision discrepancy (blocked: model/checkpoint binaries absent).
- [ ] Rerun topic modeling and regenerate final 15–20-topic LDA artifacts (blocked: source data and topic-model binaries absent).
- [ ] Browser-render pyLDAvis HTML (not verified here).
- [ ] Complete human topic review (requires actual human reviewer).
- [ ] Validate dashboard against installed Streamlit and restored models/data.
- [ ] Verify temporal/network/geographic functions with compatible real metadata, or leave them documented as blocked.
