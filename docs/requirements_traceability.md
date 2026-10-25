# Requirements Traceability Matrix

Status definitions: **Complete** = supported by source and tested here; **Partial** = implemented in part or existing artifacts exist but full behavior is not verified; **Missing** = not found in the supplied archive; **Blocked** = requires absent data, model binaries, dependencies, or human review; **Unverified** = artifact exists but behavior has not been tested.

| Requirement area | Status | Evidence / limitation |
|---|---|---|
| Clean text, tokenize, POS-aware lemmatization, extract hashtags/mentions, TF-IDF | Partial | Preprocessing module and unit tests exist; only unit tests were run in this environment. |
| Resolve 31,962/three-class vs actual dataset mismatch | Partial | Existing artifacts document 74,682 training rows, 1,000 official validation rows, and four labels. `three_exclude` and `three_merge` modes exist, but no separate three-class experiment was reproduced here. |
| EDA by class/entity, word frequencies, n-grams | Partial | Existing CSV metrics and figures are present; full pipeline was not rerun. |
| Temporal trends/anomaly detection | Blocked | Source metadata in the supplied split artifacts does not establish usable timestamps; optional compatible-data behavior not verified. |
| Retweet/mention network analysis | Blocked | No verified interaction-edge/user relationship data supplied. Mentions as text features are not a retweet network. |
| Geographic sentiment mapping | Blocked | No verified coordinates/location data supplied. |
| Logistic Regression, Random Forest, Linear SVM, Naive Bayes | Partial | Historical metrics and confusion plots exist; trained `.joblib` models are missing from the archive, so current predictions cannot be reproduced. |
| Stratified split and cross-validation | Partial | Split artifacts and CV script exist. CV was not run against source data in this environment. |
| Class imbalance strategy and confidence intervals | Partial | Historical evaluation includes macro/weighted metrics and bootstrap CI in verification artifacts; retraining/evaluation not reproduced. |
| LDA 15–20 topics | Partial | Historical topic summary uses 12 topics. Source selection was changed to choose among 15 and 20 by coherence after comparing the broader grid; regenerate topic artifacts to satisfy the requirement. |
| KMeans k=10 and NMF topic modeling | Partial | Historical artifacts exist; models are absent, and generation was not rerun. |
| pyLDAvis browser rendering | Unverified | HTML artifact exists; browser rendering not verified. |
| Human-in-the-loop topic review | Blocked | Review sheet has blank reviewer and notes columns. Human decisions must be completed by a real reviewer. |
| Fine-tuned DistilBERT | Partial / blocked | Historical metrics and logs exist; model and checkpoint binaries are missing. Stored verification JSON has metric-match failures. |
| LSTM, CNN, attention models | Partial | Deep-model source and tests exist; training/evaluation not run here. |
| Streamlit text prediction, batch upload, topic exploration, exports | Partial / blocked | `app.py` exists, but Streamlit is not installed and model artifacts are absent; dashboard smoke test could not be collected. |
| Plotly time trends and negative-spike alerts | Unverified / blocked | Requires timestamped compatible input; not validated here. |
| Reproducible source archive and instructions | Partial | Added requirements, README, traceability, audit report, and changelog. Raw data and trained model binaries are not in the supplied ZIP. |

## Dataset and label policy

The primary experiment is four-class. `three_exclude` drops `Irrelevant` examples, while `three_merge` explicitly maps them to `Neutral`; these are distinct experiments and must have separate metrics. No three-class metrics should be inferred from the four-class leaderboard.

## Selection policy

Model selection is based on internal validation macro-F1. The held-out test set is reserved for a single final evaluation and must not be used to choose models or hyperparameters. The classical leaderboard's historical selected model is Linear SVM, because it has the highest recorded validation macro-F1.
