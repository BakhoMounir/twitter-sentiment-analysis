#!/usr/bin/env python3
"""Stratified cross-validation for classical baselines on a training pool only."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_models(seed: int):
    def vec():
        return TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, strip_accents="unicode"
        )
    return {
        "LinearSVM": Pipeline([("tfidf", vec()), ("classifier", LinearSVC(random_state=seed))]),
        "LogisticRegression": Pipeline([
            ("tfidf", vec()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=seed, solver="liblinear")),
        ]),
        "ComplementNB": Pipeline([("tfidf", vec()), ("classifier", ComplementNB(alpha=0.5))]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Cleaned training pool CSV; never use held-out test data")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/cv_classical.csv")
    a = p.parse_args()
    if a.folds < 2:
        p.error("--folds must be >= 2")
    path = Path(a.input)
    if not path.exists():
        p.error(f"Input does not exist: {path}")
    df = pd.read_csv(path)
    missing = {a.text_column, a.label_column} - set(df.columns)
    if missing:
        p.error(f"Missing columns: {sorted(missing)}")
    df = df[[a.text_column, a.label_column]].dropna()
    df[a.text_column] = df[a.text_column].astype(str)
    df[a.label_column] = df[a.label_column].astype(str)
    df = df[df[a.text_column].str.strip().ne("")]
    if df[a.label_column].nunique() < 2:
        p.error("At least two classes are required")
    min_count = int(df[a.label_column].value_counts().min())
    if a.folds > min_count:
        p.error(f"folds ({a.folds}) exceeds smallest class count ({min_count})")
    x, y = df[a.text_column].tolist(), df[a.label_column].tolist()
    splitter = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    rows = []
    for name, model in build_models(a.seed).items():
        for fold, (itr, iva) in enumerate(splitter.split(x, y), 1):
            model.fit([x[i] for i in itr], [y[i] for i in itr])
            pred = model.predict([x[i] for i in iva])
            rows.append({
                "record_type": "fold", "model": name, "fold": fold,
                "n_train": len(itr), "n_validation": len(iva),
                "accuracy": accuracy_score([y[i] for i in iva], pred),
                "macro_f1": f1_score([y[i] for i in iva], pred, average="macro", zero_division=0),
                "weighted_f1": f1_score([y[i] for i in iva], pred, average="weighted", zero_division=0),
            })
            print(f"{name} fold {fold}/{a.folds}: macro-F1={rows[-1]['macro_f1']:.4f}", flush=True)
    result = pd.DataFrame(rows)
    summary = result.groupby("model", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"), weighted_f1_std=("weighted_f1", "std"),
    )
    summary["record_type"] = "summary"
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([result, summary], ignore_index=True, sort=False).to_csv(out, index=False)
    print(f"Saved CV results: {out}")


if __name__ == "__main__":
    main()
