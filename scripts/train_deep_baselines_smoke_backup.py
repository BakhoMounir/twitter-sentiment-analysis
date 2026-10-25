#!/usr/bin/env python3
"""Train simple LSTM/CNN/attention baselines from a training-pool CSV.

This is an optional standalone experiment. It uses a stratified split of the supplied CSV and
must not be given the official held-out test file.
"""
from __future__ import annotations
import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.deep_models import LSTMSentimentClassifier, MultiKernelCNNClassifier, AttentionPoolingClassifier

TOKEN_RE = re.compile(r"[\w#@']+", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts, max_vocab):
    counts = Counter(token for text in texts for token in tokenize(text))
    words = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_vocab - 2)]
    return {w: i for i, w in enumerate(words)}


def encode(text, vocab, max_length):
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens] or [vocab["<unk>"]]
    length = min(len(ids), max_length)
    ids = ids[:max_length] + [vocab["<pad>"]] * max(0, max_length - len(ids))
    return ids, length


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length, label_to_id):
        self.rows = []
        for text, label in zip(texts, labels):
            ids, length = encode(text, vocab, max_length)
            self.rows.append((ids, length, label_to_id[label]))
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        ids, length, label = self.rows[idx]
        return torch.tensor(ids), torch.tensor(length), torch.tensor(label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Cleaned training pool CSV, not the held-out test file")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    p.add_argument("--output-dir", default="results/deep_baselines")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-vocab", type=int, default=20000)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-rows", type=int, default=None)
    a = p.parse_args()
    if a.epochs < 1 or a.max_vocab < 3 or a.max_length < 1 or a.batch_size < 1:
        p.error("epochs, max length, and batch size must be positive; max vocab must be >= 3")
    frame = pd.read_csv(a.train)
    missing = {a.text_column, a.label_column} - set(frame.columns)
    if missing:
        p.error(f"Missing columns: {sorted(missing)}")
    frame = frame[[a.text_column, a.label_column]].dropna()
    frame[a.text_column] = frame[a.text_column].astype(str)
    frame[a.label_column] = frame[a.label_column].astype(str)
    frame = frame[frame[a.text_column].str.strip().ne("")]
    if a.max_train_rows and len(frame) > a.max_train_rows:
        frame, _ = train_test_split(frame, train_size=a.max_train_rows, stratify=frame[a.label_column], random_state=a.seed)
    if frame[a.label_column].nunique() < 2:
        p.error("At least two classes are required")
    train_df, val_df = train_test_split(frame, test_size=0.15, stratify=frame[a.label_column], random_state=a.seed)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = sorted(frame[a.label_column].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    vocab = build_vocab(train_df[a.text_column].tolist(), a.max_vocab)
    train_ds = TextDataset(train_df[a.text_column], train_df[a.label_column], vocab, a.max_length, label_to_id)
    val_ds = TextDataset(val_df[a.text_column], val_df[a.label_column], vocab, a.max_length, label_to_id)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False)
    factories = {
        "lstm": lambda: LSTMSentimentClassifier(len(vocab), len(labels)),
        "cnn": lambda: MultiKernelCNNClassifier(len(vocab), len(labels)),
        "attention": lambda: AttentionPoolingClassifier(len(vocab), len(labels)),
    }
    outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name, factory in factories.items():
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        best_f1, best_state, history = -1.0, None, []
        start = time.time()
        for epoch in range(1, a.epochs + 1):
            model.train(); loss_sum = 0.0
            for ids, lengths, y in train_loader:
                ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids, lengths)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += float(loss.item()) * len(y)
            model.eval(); ys, preds = [], []
            with torch.no_grad():
                for ids, lengths, y in val_loader:
                    logits = model(ids.to(device), lengths.to(device))
                    preds.extend(logits.argmax(1).cpu().tolist())
                    ys.extend(y.tolist())
            val_f1 = f1_score(ys, preds, average="macro", zero_division=0)
            history.append({"epoch": epoch, "train_loss": loss_sum / max(len(train_ds), 1),
                            "val_accuracy": accuracy_score(ys, preds), "val_macro_f1": val_f1})
            print(f"{name} epoch {epoch}/{a.epochs}: val macro-F1={val_f1:.4f}", flush=True)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval(); ys, preds = [], []
        with torch.no_grad():
            for ids, lengths, y in val_loader:
                preds.extend(model(ids.to(device), lengths.to(device)).argmax(1).cpu().tolist())
                ys.extend(y.tolist())
        result = {
            "model": name, "validation_rows": len(val_ds), "accuracy": float(accuracy_score(ys, preds)),
            "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(ys, preds, average="weighted", zero_division=0)),
            "training_seconds": round(time.time() - start, 2), "device": str(device),
            "labels": labels, "classification_report": classification_report(
                ys, preds, labels=list(range(len(labels))), output_dict=True, zero_division=0
            ),
        }
        torch.save({"state_dict": best_state, "vocab": vocab, "labels": labels,
                    "max_length": a.max_length, "model": name}, outdir / f"{name}.pt")
        (outdir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
        pd.DataFrame(history).to_csv(outdir / f"{name}_history.csv", index=False)
        all_metrics.append({k: v for k, v in result.items() if k != "classification_report"})
    pd.DataFrame(all_metrics).to_csv(outdir / "summary.csv", index=False)
    print(f"Saved baseline artifacts to {outdir}")


if __name__ == "__main__":
    main()
