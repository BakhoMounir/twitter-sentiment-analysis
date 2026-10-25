"""DistilBERT fine-tuning (Hugging Face + PyTorch). Requires: torch, transformers, accelerate.

Protocol: train on `train`, early-stop / pick checkpoint on `val` (macro-F1), report `test` once.
Input is minimally normalised text (case, punctuation, emojis and negation preserved).
Label ids follow config.labels_for_mode(mode) so training and inference always agree.
"""
from __future__ import annotations

import inspect
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEED, Paths, labels_for_mode
from .data import load_splits
from .evaluation import evaluate, plot_confusion, save_json, upsert_comparison
from .preprocessing import normalize_minimal

MODEL_NAME = os.environ.get("TSP_TRANSFORMER", "distilbert-base-uncased")


def run_transformer(paths: Paths, mode: str, epochs: int = 3, lr: float = 2e-5, batch: int = 16,
                    max_len: int = 64, max_train: int | None = None, patience: int = 1, log=print) -> dict:
    import torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
    set_seed(SEED)
    labels = labels_for_mode(mode); l2i = {l: i for i, l in enumerate(labels)}
    sp = load_splits(paths)
    tr = sp["train"]
    if max_train and len(tr) > max_train:
        tr = tr.sample(max_train, random_state=SEED)          # stratification is approximate; documented in report
        log(f"  training on a {max_train:,}-row subsample (CPU budget)")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    class DS(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tok([normalize_minimal(t) for t in df.text], truncation=True, max_length=max_len)
            self.y = [l2i[l] for l in df.label]
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            d = {k: v[i] for k, v in self.enc.items()}; d["labels"] = self.y[i]; return d

    dtr, dva, dte = DS(tr), DS(sp["val"]), DS(sp["test"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"accuracy": float((pred == p.label_ids).mean()),
                "macro_f1": float(f1_score(p.label_ids, pred, average="macro"))}

    valid = inspect.signature(TrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in valid else "evaluation_strategy"
    total_steps = -(-len(dtr) // batch) * epochs
    kw = dict(
        output_dir=str(paths.models / "distilbert_ckpt"), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch, per_device_eval_batch_size=64, weight_decay=0.01,
        warmup_steps=int(0.06 * total_steps),            # int steps: works in transformers 4.x and 5.x
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, logging_steps=50, seed=SEED, data_seed=SEED, report_to="none",
        fp16=torch.cuda.is_available(), **{key: "epoch"})
    dropped = [k for k in kw if k not in valid]
    if dropped:
        log(f"  note: TrainingArguments in this transformers version has no {dropped}; ignored")
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in valid})
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    t0 = time.time(); trainer.train(); train_s = time.time() - t0

    def predict(ds):
        t = time.time(); out = trainer.predict(ds); return out.predictions.argmax(-1), time.time() - t
    pv, _ = predict(dva); pt, inf_s = predict(dte)
    yv = [labels[i] for i in dva.y]; yt = [labels[i] for i in dte.y]
    mv = evaluate(yv, [labels[i] for i in pv], labels); mt = evaluate(yt, [labels[i] for i in pt], labels)

    out = paths.models / "distilbert"
    trainer.save_model(str(out)); tok.save_pretrained(str(out))
    save_json({"labels": labels}, out / "label_map.json")
    env = {"model": MODEL_NAME, "torch": torch.__version__, "device": "cuda:" + torch.cuda.get_device_name(0)
           if torch.cuda.is_available() else "cpu", "python": platform.python_version(),
           "epochs_requested": epochs, "lr": lr, "batch": batch, "max_len": max_len,
           "train_rows": len(dtr), "train_seconds": round(train_s, 1)}
    mt.update(model="distilbert", val=mv, environment=env, val_macro_f1=mv["macro_f1"])
    save_json(mt, paths.metrics / "transformer_distilbert.json")
    plot_confusion(mt["confusion_matrix"], labels, "DistilBERT - test (official validation file)", paths.figures / "confusion_distilbert.png")

    hist = pd.DataFrame(trainer.state.log_history); hist.to_csv(paths.metrics / "transformer_training_log.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    if "loss" in hist: hist.dropna(subset=["loss"]).plot(x="step", y="loss", ax=ax[0], title="Training loss", legend=False)
    if "eval_macro_f1" in hist: hist.dropna(subset=["eval_macro_f1"]).plot(x="epoch", y=["eval_macro_f1", "eval_accuracy"], marker="o", ax=ax[1], title="Validation metrics")
    fig.tight_layout(); fig.savefig(paths.figures / "transformer_learning_curve.png", dpi=130); plt.close(fig)

    upsert_comparison(paths, {"model": "distilbert", "family": "transformer", "params": json.dumps({"epochs": epochs, "lr": lr, "max_len": max_len}),
                              "n_features": None, "val_macro_f1": mv["macro_f1"], "test_accuracy": mt["accuracy"],
                              "test_macro_f1": mt["macro_f1"], "test_weighted_f1": mt["weighted_f1"],
                              "train_seconds": round(train_s, 1), "inference_ms_per_1k": round(inf_s / len(dte) * 1e6, 1)})
    log(f"  DistilBERT TEST acc={mt['accuracy']:.4f} macro-F1={mt['macro_f1']:.4f} ({train_s/60:.1f} min on {env['device']})")
    return mt


def predict_texts(texts: list[str], model_dir: Path, labels: list[str], max_len: int = 64, batch: int = 32) -> np.ndarray:
    """Softmax probabilities in `labels` order; the label map stored with the model is checked."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    saved = json.loads((Path(model_dir) / "label_map.json").read_text())["labels"]
    if saved != labels:
        raise ValueError(f"Label mismatch between model {saved} and project {labels}")
    tok = AutoTokenizer.from_pretrained(str(model_dir)); model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok([normalize_minimal(t) for t in texts[i:i + batch]], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1).numpy())
    return np.vstack(probs)
