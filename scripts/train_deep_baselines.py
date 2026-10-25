
#!/usr/bin/env python3
"""Train LSTM/CNN/attention baselines using fixed train/val/test CSV splits.

The validation set selects the best epoch. The held-out test set is evaluated
only after selection and is never used for training or model selection.
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.deep_models import (
    LSTMSentimentClassifier,
    MultiKernelCNNClassifier,
    AttentionPoolingClassifier,
)

TOKEN_RE = re.compile(r"[\w#@']+", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts, max_vocab):
    counts = Counter(token for text in texts for token in tokenize(text))
    words = ["<pad>", "<unk>"] + [
        word for word, _ in counts.most_common(max_vocab - 2)
    ]
    return {word: i for i, word in enumerate(words)}


def encode(text, vocab, max_length):
    tokens = tokenize(text)[:max_length]
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    if not ids:
        ids = [vocab["<unk>"]]
    length = min(len(ids), max_length)
    ids = ids[:max_length]
    ids += [vocab["<pad>"]] * (max_length - len(ids))
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


def read_split(path, text_column, label_column):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    frame = frame[[text_column, label_column]].copy()
    frame[text_column] = frame[text_column].astype(str).str.strip()
    frame[label_column] = frame[label_column].astype(str).str.strip()
    frame = frame[
        frame[text_column].ne("") & frame[label_column].ne("")
    ].reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"{path}: no usable rows")

    return frame


def evaluate(model, loader, device, labels):
    model.eval()
    actual, predicted = [], []

    with torch.no_grad():
        for ids, lengths, y in loader:
            logits = model(ids.to(device), lengths.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(y.tolist())

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "rows": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(actual, predicted, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(range(len(labels)))
        ).tolist(),
        "actual_ids": actual,
        "predicted_ids": predicted,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train deep sentiment baselines with fixed data splits."
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-dir", default="results/four/deep_baselines")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()

    if (
        args.epochs < 1
        or args.max_vocab < 3
        or args.max_length < 1
        or args.batch_size < 1
        or (args.max_train_rows is not None and args.max_train_rows < 1)
    ):
        parser.error("epochs, max length, batch size, and any row limit must be positive; max vocab must be >= 3")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_df = read_split(args.train, args.text_column, args.label_column)
    val_df = read_split(args.val, args.text_column, args.label_column)
    test_df = read_split(args.test, args.text_column, args.label_column)

    # Optional development subsampling applies to training only.
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        train_df, _ = train_test_split(
            train_df,
            train_size=args.max_train_rows,
            stratify=train_df[args.label_column],
            random_state=args.seed,
        )
        train_df = train_df.reset_index(drop=True)

    labels = sorted(train_df[args.label_column].unique().tolist())
    label_set = set(labels)

    for name, frame in (("validation", val_df), ("test", test_df)):
        unknown = set(frame[args.label_column].unique()) - label_set
        if unknown:
            parser.error(f"{name} split contains labels absent from training: {sorted(unknown)}")

    label_to_id = {label: i for i, label in enumerate(labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Labels (ID order): {label_to_id}")
    print(
        f"Rows: train={len(train_df)}, val={len(val_df)}, "
        f"test={len(test_df)}"
    )

    # Fit vocabulary on training texts only.
    vocab = build_vocab(train_df[args.text_column].tolist(), args.max_vocab)

    train_ds = TextDataset(
        train_df[args.text_column], train_df[args.label_column],
        vocab, args.max_length, label_to_id,
    )
    val_ds = TextDataset(
        val_df[args.text_column], val_df[args.label_column],
        vocab, args.max_length, label_to_id,
    )
    test_ds = TextDataset(
        test_df[args.text_column], test_df[args.label_column],
        vocab, args.max_length, label_to_id,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    factories = {
        "lstm": lambda: LSTMSentimentClassifier(len(vocab), len(labels)),
        "cnn": lambda: MultiKernelCNNClassifier(len(vocab), len(labels)),
        "attention": lambda: AttentionPoolingClassifier(len(vocab), len(labels)),
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, factory in factories.items():
        # Reset initialization seed so the experiment is reproducible.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_f1 = -1.0
        best_epoch = 0
        best_state = None
        history = []
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0

            for ids, lengths, y in train_loader:
                ids = ids.to(device)
                lengths = lengths.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(ids, lengths)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(y)

            val_metrics = evaluate(model, val_loader, device, labels)
            val_f1 = val_metrics["macro_f1"]

            history.append({
                "epoch": epoch,
                "train_loss": loss_sum / max(len(train_ds), 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1,
                "val_weighted_f1": val_metrics["weighted_f1"],
            })

            print(
                f"{name} epoch {epoch}/{args.epochs}: "
                f"val macro-F1={val_f1:.4f}",
                flush=True,
            )

            # Select checkpoint using validation data only.
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        model.load_state_dict(best_state)
        model.to(device)

        # Test is used only after the best epoch has been selected.
        test_metrics = evaluate(model, test_loader, device, labels)

        checkpoint = {
            "state_dict": best_state,
            "vocab": vocab,
            "labels": labels,
            "label_to_id": label_to_id,
            "max_length": args.max_length,
            "model": name,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "seed": args.seed,
        }
        torch.save(checkpoint, outdir / f"{name}.pt")

        # Save test predictions with original sentiment names.
        predictions = pd.DataFrame({
            "text": test_df[args.text_column].tolist(),
            "true_label": [
                labels[i] for i in test_metrics["actual_ids"]
            ],
            "predicted_label": [
                labels[i] for i in test_metrics["predicted_ids"]
            ],
        })
        predictions.to_csv(outdir / f"{name}_test_predictions.csv", index=False)

        result = {
            "model": name,
            "best_epoch": best_epoch,
            "train_rows": len(train_ds),
            "validation_rows": len(val_ds),
            "test_rows": test_metrics["rows"],
            "validation_macro_f1": best_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "training_seconds": round(time.time() - start, 2),
            "device": str(device),
            "labels": labels,
            "label_to_id": label_to_id,
            "test_classification_report": test_metrics["classification_report"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
        }

        (outdir / f"{name}_metrics.json").write_text(
            json.dumps(result, indent=2)
        )
        pd.DataFrame(history).to_csv(
            outdir / f"{name}_history.csv", index=False
        )

        summary.append({
            key: value for key, value in result.items()
            if key not in ("test_classification_report", "test_confusion_matrix")
        })

        print(
            f"{name}: best epoch={best_epoch}, "
            f"validation macro-F1={best_f1:.4f}, "
            f"test macro-F1={test_metrics['macro_f1']:.4f}",
            flush=True,
        )

    pd.DataFrame(summary).to_csv(outdir / "summary.csv", index=False)
    (outdir / "experiment_config.json").write_text(json.dumps({
        "train_file": args.train,
        "validation_file": args.val,
        "test_file": args.test,
        "epochs": args.epochs,
        "max_vocab": args.max_vocab,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "labels": labels,
        "label_to_id": label_to_id,
        "test_used_for_selection": False,
    }, indent=2))

    print(f"Saved baseline artifacts to {outdir}")


if __name__ == "__main__":
    main()
