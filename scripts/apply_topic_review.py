#!/usr/bin/env python3
"""Clean topic-term lists and create a human-review template; does not auto-label topics."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
DOMAIN_RE = re.compile(r"\b(?:t\.co|twitter\.com|youtu\.be|youtube\.com|pic\.twitter)\b", re.I)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]*", re.I)
NOISE = {"unk", "amp", "http", "https", "www", "com", "pic", "twitter", "youtu", "youtube", "tco", "rt"}


def clean_terms(value, extra_stopwords=None):
    if pd.isna(value):
        return []
    text = DOMAIN_RE.sub(" ", URL_RE.sub(" ", str(value).lower()))
    stop = NOISE | (extra_stopwords or set())
    output, seen = [], set()
    for token in TOKEN_RE.findall(text):
        token = token.strip("_-'")
        if len(token) < 2 or token in stop or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--topic-column", default="topic_id")
    p.add_argument("--terms-column", default=None)
    p.add_argument("--stopwords", default=None, help="Optional newline-separated extra stopwords")
    a = p.parse_args()
    source = Path(a.input)
    if not source.exists():
        p.error(f"Input file does not exist: {source}")
    df = pd.read_csv(source)
    if a.topic_column not in df.columns:
        p.error(f"Missing topic column {a.topic_column!r}")
    terms_col = a.terms_column or next((c for c in ("terms", "top_terms", "keywords") if c in df.columns), None)
    if terms_col not in df.columns:
        p.error("Terms column not found; specify --terms-column")
    extra = set()
    if a.stopwords:
        extra = {s.strip().lower() for s in Path(a.stopwords).read_text().splitlines()
                 if s.strip() and not s.lstrip().startswith("#")}
    out_df = df.copy()
    out_df["cleaned_terms"] = out_df[terms_col].apply(lambda v: ", ".join(clean_terms(v, extra)))
    out_df["human_topic_label"] = ""
    out_df["reviewer_notes"] = ""
    out_df["review_status"] = "needs_review"
    out_df["topic_label_source"] = "human_review_required"
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} topic rows to {out}; labels remain unreviewed.")


if __name__ == "__main__":
    main()
