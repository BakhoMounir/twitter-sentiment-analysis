#!/usr/bin/env python3
"""Compare reference and reloaded-model prediction exports row by row."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report


def metrics(df, label_col, pred_col):
    y = df[label_col].astype(str)
    pred = df[pred_col].astype(str)
    labels = sorted(set(y) | set(pred))
    return {
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--reloaded", required=True)
    p.add_argument("--label-column", default="y_true")
    p.add_argument("--prediction-column", default="y_pred")
    p.add_argument("--id-column", default=None)
    p.add_argument("--output", default="results/transformer/diagnostic.json")
    a = p.parse_args()
    rp, npth = Path(a.reference), Path(a.reloaded)
    if not rp.exists() or not npth.exists():
        p.error("Both prediction CSVs must exist")
    ref, new = pd.read_csv(rp), pd.read_csv(npth)
    required = {a.label_column, a.prediction_column}
    for name, df in (("reference", ref), ("reloaded", new)):
        missing = required - set(df.columns)
        if missing:
            p.error(f"{name} CSV missing columns: {sorted(missing)}")
    if len(ref) != len(new):
        p.error(f"Row count mismatch: reference={len(ref)}, reloaded={len(new)}")
    id_mismatches = None
    if a.id_column:
        if a.id_column not in ref or a.id_column not in new:
            p.error(f"ID column {a.id_column!r} must be in both files")
        id_mismatches = int((ref[a.id_column].astype(str).to_numpy() != new[a.id_column].astype(str).to_numpy()).sum())
    true_mismatches = int((ref[a.label_column].astype(str).to_numpy() != new[a.label_column].astype(str).to_numpy()).sum())
    ref_pred, new_pred = ref[a.prediction_column].astype(str).to_numpy(), new[a.prediction_column].astype(str).to_numpy()
    mismatch_idx = np.flatnonzero(ref_pred != new_pred)
    result = {
        "reference_file": str(rp), "reloaded_file": str(npth), "row_count": int(len(ref)),
        "id_order_mismatches": id_mismatches, "true_label_mismatches": true_mismatches,
        "prediction_mismatches": int(len(mismatch_idx)),
        "mismatch_examples": [
            {"row": int(i), "reference_prediction": str(ref_pred[i]), "reloaded_prediction": str(new_pred[i])}
            for i in mismatch_idx[:100]
        ],
        "reference_metrics": metrics(ref, a.label_column, a.prediction_column),
        "reloaded_metrics": metrics(new, a.label_column, a.prediction_column),
        "interpretation": (
            "Sample IDs are not aligned; fix ordering before interpreting metric differences."
            if id_mismatches not in (None, 0) else
            "Reference and reloaded predictions differ; investigate preprocessing, checkpoint, max length, "
            "label-index mapping, precision, and model evaluation mode."
            if len(mismatch_idx) else "Predictions match row by row."
        ),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("reference_metrics", "reloaded_metrics", "mismatch_examples")}, indent=2))
    print(f"Saved full diagnostic: {out}")


if __name__ == "__main__":
    main()
