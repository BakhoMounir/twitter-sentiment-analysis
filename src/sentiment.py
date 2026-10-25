"""Unified inference for saved classical and Transformer models (same label order everywhere)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .config import Paths


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def get_labels(paths: Paths) -> list[str]:
    return json.loads((paths.models / "label_map.json").read_text())["labels"]


def available_models(paths: Paths) -> dict[str, str]:
    """name -> kind ('classical' | 'transformer') for artifacts that really exist."""
    found = {p.stem: "classical" for p in sorted((paths.models / "classical").glob("*.joblib"))}
    if (paths.models / "distilbert" / "config.json").exists():
        found["distilbert"] = "transformer"
    return found


def load_classical(paths: Paths, name: str):
    return joblib.load(paths.models / "classical" / f"{name}.joblib")


def predict_classical(pipe, texts: list[str], labels: list[str]):
    """Returns (predicted labels, probability-like matrix in `labels` order, kind)."""
    clf = pipe.steps[-1][1]
    if hasattr(pipe, "predict_proba"):
        P, kind = pipe.predict_proba(texts), "probability"
    else:
        P, kind = _softmax(np.atleast_2d(pipe.decision_function(texts))), "pseudo-probability (softmax of SVM margin)"
    order = [list(clf.classes_).index(l) for l in labels]
    P = P[:, order]
    return [labels[i] for i in P.argmax(1)], P, kind


def predict_transformer(paths: Paths, texts: list[str], labels: list[str], max_len: int = 64):
    from .transformer import predict_texts  # lazy: torch is optional
    P = predict_texts(texts, paths.models / "distilbert", labels, max_len=max_len)
    return [labels[i] for i in P.argmax(1)], P, "probability"
