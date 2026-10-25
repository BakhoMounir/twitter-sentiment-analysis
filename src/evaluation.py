"""Metrics, confusion matrices and the model-comparison table."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import Paths


def _conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, Path): return str(o)
    raise TypeError(type(o))


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_conv))


def evaluate(y_true, y_pred, labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, digits=4,
                                                       zero_division=0),
        "labels": labels,
    }


def plot_confusion(cm, labels, title: str, path: Path) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Counts"), (axes[1], norm, ".2f", "Row-normalised (recall)")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sub)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def upsert_comparison(paths: Paths, row: dict) -> pd.DataFrame:
    """Add/replace a model row; the selected model is the one with highest VALIDATION macro-F1."""
    f = paths.metrics / "model_comparison.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if len(df):
        df = df[df["model"] != row["model"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
    df["selected_by_val_macro_f1"] = False
    if df["val_macro_f1"].notna().any():
        df.loc[df["val_macro_f1"].idxmax(), "selected_by_val_macro_f1"] = True
    df.to_csv(f, index=False)
    return df
