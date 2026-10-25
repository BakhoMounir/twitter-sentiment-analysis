"""Builds results/<mode>/reports/final_report.md strictly from metrics files that exist."""
from __future__ import annotations

import json

import pandas as pd

from .config import Paths


def _j(p):
    return json.loads(p.read_text()) if p.exists() else None


def build_report(paths: Paths, mode: str) -> str:
    dv = _j(paths.metrics / "data_validation.json"); env = _j(paths.metrics / "environment.json")
    cmp_f = paths.metrics / "model_comparison.csv"; ts = _j(paths.topics / "topic_summary.json")
    L = [f"# Twitter Sentiment & Topic Project - Results Report (label mode: `{mode}`)\n",
         "_Generated automatically from files in `results/`; every number below was produced by the pipeline run that created them._\n"]
    if dv:
        s = dv["splits"]; c = dv["cleaning"]
        L += ["## 1. Data and evaluation protocol",
              f"- Training file rows read: **{dv['training_file']['rows_read']:,}**; validation file rows read: **{dv['validation_file']['rows_read']:,}** "
              f"(header detected: {dv['training_file']['header_detected']}).",
              f"- Unmapped-label rows (training/validation): {dv['training_file']['unmapped_label_rows']} / {dv['validation_file']['unmapped_label_rows']} (dropped, never mapped to Neutral).",
              f"- Training cleaning: {c['train_rows_dropped_conflicting_labels']} rows dropped for conflicting labels, "
              f"{c['train_rows_dropped_duplicate_text']} duplicate texts removed, "
              f"{c['train_rows_dropped_overlap_with_official_validation']} rows removed for overlap with the official validation file.",
              f"- Label set: {', '.join(s['labels'])}.", f"- Protocol: {s['protocol']}.",
              f"- Rows - train {s['train']['rows']:,}, val {s['val']['rows']:,}, test {s['test']['rows']:,}.", ""]
    if cmp_f.exists():
        df = pd.read_csv(cmp_f)
        cols = [c for c in ["model", "val_macro_f1", "test_accuracy", "test_macro_f1",
                            "test_weighted_f1", "train_seconds", "inference_ms_per_1k"] if c in df.columns]
        L += ["## 2. Training-time model comparison",
              "These are metrics recorded during the training pipeline; they are not a substitute for the independent saved-artifact verification below. "
              "Model selection must use validation macro-F1 only.\n",
              df[cols].round(4).to_markdown(index=False), ""]
        if "selected_by_val_macro_f1" in df.columns:
            selected = df["selected_by_val_macro_f1"].astype(str).str.lower().isin(["true", "1", "yes"])
            if selected.any():
                best = df.loc[selected].iloc[0]
                L += [f"Training-stage selection record: `{best['model']}` (validation macro-F1 {float(best['val_macro_f1']):.4f}).", ""]
            else:
                L += ["> No selected model is marked in `model_comparison.csv`; selection status needs review.", ""]
        if "distilbert" not in set(df["model"]):
            L += ["> DistilBERT results are absent from this training-time comparison; the transformer stage may not have been run for this output.", ""]
    nd = _j(paths.metrics / "near_duplicate_diagnostic.json")
    if nd:
        t = nd["test"]
        L += ["## 2b. Near-duplicate / leakage diagnostic",
              f"- {t['frac_near_duplicate_of_train']:.1%} of held-out test tweets have a training tweet with cosine similarity >= {nd['threshold_cosine']} "
              f"(validation sample: {nd['val_sample']['frac_near_duplicate_of_train']:.1%}).",
              f"- `{nd['model_used']}` accuracy on those near-duplicates: {t['accuracy_on_near_duplicates']}; on novel tweets: {t['accuracy_on_novel']}.",
              "- A large gap means headline scores partly measure memorisation of repeated tweets; report the *novel* figure as the conservative estimate.", ""]
    lbf = paths.metrics / "leaderboard.csv"
    if lbf.exists():
        lb = pd.read_csv(lbf)
        cols = [c for c in [
            "model", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_macro_f1_ci95_low",
            "test_macro_f1_ci95_high", "test_weighted_f1", "train_seconds",
            "inference_ms_per_1k_tweets", "model_size_mb", "device_for_inference", "all_checks_passed"
        ] if c in lb.columns]
        L += ["## 2c. Verified leaderboard (saved models re-evaluated from disk)",
              "val_* = internal validation split (selection); test_* = official validation file (held out). "
              "CI = bootstrap 95% over test tweets; inference time is end-to-end from raw text on the listed device.\\n",
              lb[cols].round(4).to_markdown(index=False), ""]
        selected_col = "selected_by_val_macro_f1"
        if selected_col in lb.columns:
            selected_mask = lb[selected_col].astype(str).str.lower().isin(["true", "1", "yes"])
            if selected_mask.any():
                selected_row = lb.loc[selected_mask].iloc[0]
                L += [f"**Verified leaderboard selection:** `{selected_row['model']}` selected by validation macro-F1 "
                      f"({float(selected_row['val_macro_f1']):.4f}); held-out test scores are not used for selection.", ""]
        failed = []
        if "all_checks_passed" in lb.columns:
            passed = lb["all_checks_passed"].astype(str).str.lower().isin(["true", "1", "yes"])
            failed = lb.loc[~passed, "model"].astype(str).tolist()
        if failed:
            L += ["> **Verification warning:** checks did not all pass for: " + ", ".join(f"`{m}`" for m in failed) +
                  ". Do not present these saved-model metrics as fully verified until the mismatch is resolved. "
                  f"See `results/{mode}/metrics/verification.json` for individual checks.", ""]
        nd2 = paths.metrics / "near_duplicate_by_model.csv"
        if nd2.exists():
            L += ["Test accuracy by maximum TF-IDF cosine similarity to any training tweet:\\n",
                  pd.read_csv(nd2).pivot(index="model", columns="similarity_bin", values="accuracy").round(4).to_markdown(), ""]

    pc = paths.metrics / "preprocessing_comparison.csv"
    if pc.exists():
        L += ["## 3. Preprocessing alternatives (LogReg, validation split)", pd.read_csv(pc).round(4).to_markdown(index=False), ""]
    if ts:
        L += ["## 4. Topic discovery (fit on training text only)"]
        for m, d in ts["methods"].items():
            L.append(f"### {m.upper()} - k = {d['k']}\n_{d['k_rule']}_\n")
            for t in d["topics"]:
                neg = t["sentiment_distribution"].get("Negative", 0)
                L.append(f"- **{m} {t['id']}** ({t['size']:,} tweets, {neg:.0%} negative): {', '.join(t['terms'][:8])}")
            L.append("")
        L += ["Topic labels are machine-generated term lists. No human validation is claimed unless "
              "`results/<mode>/topics/topic_labels.json` has been filled in by a reviewer.", ""]
    if env:
        L += ["## 5. Environment", "```json", json.dumps(env, indent=2), "```", ""]
    L += ["## 6. Limitations",
          "- The entity dataset has no timestamps, coordinates or retweet links: no temporal, geographic or network analysis is performed.",
          "- Tweets are short and often topic-redundant; topic models and clusters are descriptive, not proof of semantic categories.",
          "- Cross-validation was not used (fixed stratified validation split for speed); differences of a few tenths of a point are within noise.",
          "- Deep-learning scope: DistilBERT only (BiLSTM/CNN not implemented)."]
    txt = "\n".join(L)
    (paths.reports / "final_report.md").write_text(txt)
    return txt