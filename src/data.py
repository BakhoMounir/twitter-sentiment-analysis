"""Data loading, validation, label policy and leakage-safe splitting."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, RAW_LABELS, SEED, Paths, labels_for_mode

COLS = ["id", "entity", "sentiment", "text"]
_CANON = {l.lower(): l for l in RAW_LABELS}


def text_key(s: pd.Series) -> pd.Series:
    """Strict key for duplicate / overlap detection: ignores case, URLs, @mentions, punctuation, spaces
    (catches near-copies such as 'Great game!' vs 'great game'). Falls back to the lowercase text when the
    stripped key would be empty (e.g. emoji-only tweets)."""
    low = s.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    key = low.str.replace(r"https?://\S+|www\.\S+|@\w+", "", regex=True).str.replace(r"[\W_]+", "", regex=True)
    return key.where(key != "", low)


def _inspect(path: Path) -> tuple[bool, dict]:
    """Detect a header row and count fields per row (finds malformed rows)."""
    has_header, counts = False, Counter()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0 and len(row) >= 3 and row[2].strip().lower() in ("sentiment", "label"):
                has_header = True
            counts[len(row)] += 1
    return has_header, {str(k): v for k, v in sorted(counts.items())}


def read_raw(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read a (normally headerless) file with explicit column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put twitter_training.csv and twitter_validation.csv in "
            f"{DATA_DIR} or set TSP_DATA_DIR.")
    has_header, field_counts = _inspect(path)
    df = pd.read_csv(path, header=0 if has_header else None, names=COLS, dtype=str,
                     keep_default_na=False, on_bad_lines="skip", encoding_errors="replace")
    df["text"] = df["text"].str.strip().replace("", pd.NA)
    df["sentiment_raw"] = df["sentiment"]
    df["sentiment"] = df["sentiment"].str.strip().str.lower().map(_CANON)  # NaN if unknown
    meta = {"file": path.name, "header_detected": has_header, "fields_per_row": field_counts,
            "rows_read": int(len(df))}
    return df, meta


def validate(df: pd.DataFrame, meta: dict) -> dict:
    unknown = df.loc[df["sentiment"].isna(), "sentiment_raw"].value_counts().to_dict()
    key = text_key(df["text"])
    rep = dict(meta)
    rep.update({
        "missing_or_empty_text": int(df["text"].isna().sum()),
        "unmapped_label_rows": int(df["sentiment"].isna().sum()),
        "unmapped_labels": {str(k): int(v) for k, v in unknown.items()},
        "exact_duplicate_rows": int(df.duplicated(subset=["entity", "sentiment", "text"]).sum()),
        "duplicate_texts": int(key[df["text"].notna()].duplicated().sum()),
        "n_entities": int(df["entity"].nunique()),
        "label_counts": {k: int(v) for k, v in df["sentiment"].value_counts(dropna=True).items()},
    })
    return rep


def apply_label_policy(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()
    if mode == "three_exclude":
        df = df[df["sentiment"] != "Irrelevant"]
    elif mode == "three_merge":
        df["sentiment"] = df["sentiment"].replace({"Irrelevant": "Neutral"})
    df["label"] = df["sentiment"]
    return df


def prepare_splits(paths: Paths, mode: str, val_size: float = 0.15) -> dict:
    """Official training file -> train/val (model selection). Official validation file -> test.

    * rows with empty text or UNKNOWN labels are removed and reported (never mapped to Neutral)
    * training texts that occur with conflicting labels are dropped; repeated texts are de-duplicated
    * training texts that also appear in the official validation file are removed from training
    * the official validation file is only cleaned of invalid rows (kept otherwise intact)
    """
    tr_raw, tr_meta = read_raw(DATA_DIR / "twitter_training.csv")
    te_raw, te_meta = read_raw(DATA_DIR / "twitter_validation.csv")
    report = {"label_mode": mode, "seed": SEED,
              "training_file": validate(tr_raw, tr_meta), "validation_file": validate(te_raw, te_meta)}

    def drop_invalid(df):
        return df[df["text"].notna() & df["sentiment"].notna()].copy()

    tr, te = drop_invalid(tr_raw), drop_invalid(te_raw)
    tr["key"], te["key"] = text_key(tr["text"]), text_key(te["text"])

    n0 = len(tr)
    nun = tr.groupby("key")["sentiment"].transform("nunique")
    tr = tr[nun == 1]
    n_conflict = n0 - len(tr)
    n1 = len(tr)
    tr = tr.drop_duplicates(subset="key", keep="first")
    n_dup = n1 - len(tr)
    overlap = tr["key"].isin(set(te["key"]))
    n_overlap = int(overlap.sum())
    tr = tr[~overlap]

    tr, te = apply_label_policy(tr, mode), apply_label_policy(te, mode)
    labels = labels_for_mode(mode)
    tr_fit, va = train_test_split(tr, test_size=val_size, stratify=tr["label"], random_state=SEED)

    report["cleaning"] = {
        "train_rows_after_invalid_removal": int(n0),
        "train_rows_dropped_conflicting_labels": int(n_conflict),
        "train_rows_dropped_duplicate_text": int(n_dup),
        "train_rows_dropped_overlap_with_official_validation": n_overlap,
        "test_duplicate_texts_kept": int(te["key"].duplicated().sum()),
    }
    report["splits"] = {
        "protocol": ("official training file -> stratified %d/%d train/val (model selection); "
                     "official validation file = held-out test (never used for selection)"
                     % (round((1 - val_size) * 100), round(val_size * 100))),
        "labels": labels,
        **{n: {"rows": int(len(d)), "label_counts": {l: int((d['label'] == l).sum()) for l in labels}}
           for n, d in (("train", tr_fit), ("val", va), ("test", te))},
    }
    cols = ["id", "entity", "sentiment", "label", "text"]
    for name, d in (("train", tr_fit), ("val", va), ("test", te)):
        d[cols].to_csv(paths.splits / f"{name}.csv", index=False)
    (paths.metrics / "data_validation.json").write_text(json.dumps(report, indent=2))
    return report


def load_splits(paths: Paths) -> dict[str, pd.DataFrame]:
    out = {}
    for n in ("train", "val", "test"):
        f = paths.splits / f"{n}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run: python scripts/run_all.py --stages data")
        out[n] = pd.read_csv(f, dtype={"id": str}, keep_default_na=False)
    return out
