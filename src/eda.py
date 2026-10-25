"""Exploratory data analysis on the TRAIN+VAL splits (the held-out test file is not inspected)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from .config import Paths, labels_for_mode
from .data import load_splits
from .preprocessing import extract_entities

PAL = {"Negative": "#d1495b", "Neutral": "#8d99ae", "Positive": "#2a9d8f", "Irrelevant": "#edae49"}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _top_ngrams(df, labels, n, k, stop):
    out = {}
    for lab in labels:
        txt = df.loc[df.label == lab, "text"].str.lower().str.replace(r"https?://\S+|@\w+", " ", regex=True)
        cv = CountVectorizer(ngram_range=(n, n), stop_words=list(ENGLISH_STOP_WORDS) if stop else None,
                             min_df=3, max_features=50_000, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
        try:
            M = cv.fit_transform(txt)
        except ValueError:
            out[lab] = pd.Series(dtype=float); continue
        s = pd.Series(M.sum(axis=0).A1, index=cv.get_feature_names_out()).nlargest(k)
        out[lab] = s
    return out


def run_eda(paths: Paths, mode: str, log=print) -> None:
    labels = labels_for_mode(mode)
    sp = load_splits(paths)
    df = pd.concat([sp["train"], sp["val"]], ignore_index=True)
    df["n_chars"] = df.text.str.len()
    df["n_tokens"] = df.text.str.split().str.len()
    ents = df.text.map(extract_entities)
    df["n_mentions"] = ents.map(lambda x: len(x[0])); df["n_hashtags"] = ents.map(lambda x: len(x[1]))
    pal = {l: PAL[l] for l in labels}
    F = paths.figures
    sns.set_theme(style="whitegrid")

    # class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="label", order=labels, palette=pal, hue="label", legend=False, ax=ax)
    for c in ax.containers: ax.bar_label(c)
    ax.set_title("Sentiment class distribution (train+val)"); ax.set_xlabel("")
    _save(fig, F / "eda_class_distribution.png")

    # entity x sentiment
    ct = pd.crosstab(df.entity, df.label).reindex(columns=labels, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    prop = ct[labels].div(ct["total"], axis=0)
    ct.join(prop.add_suffix("_share")).sort_values("total", ascending=False).to_csv(paths.metrics / "entity_sentiment.csv")
    top = ct.sort_values("total", ascending=False).head(25).index
    fig, ax = plt.subplots(figsize=(8, 8))
    prop.loc[top[::-1]].plot(kind="barh", stacked=True, color=[pal[l] for l in labels], ax=ax, width=0.85)
    ax.set_title("Sentiment mix of the 25 most frequent entities"); ax.set_xlabel("share of tweets"); ax.set_ylabel("")
    ax.legend(bbox_to_anchor=(1, 1))
    _save(fig, F / "eda_entity_sentiment_share.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    ct["total"].sort_values().tail(25).plot(kind="barh", color="#457b9d", ax=ax)
    ax.set_title("Tweet count of the 25 most frequent entities"); ax.set_ylabel("")
    _save(fig, F / "eda_top_entities.png")

    # length
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=df, x="n_tokens", hue="label", hue_order=labels, palette=pal, bins=40, element="step",
                 stat="density", common_norm=False, ax=axes[0]); axes[0].set_title("Whitespace tokens per tweet")
    sns.boxplot(data=df, x="label", y="n_chars", order=labels, palette=pal, hue="label", legend=False, ax=axes[1],
                showfliers=False); axes[1].set_title("Characters per tweet"); axes[1].set_xlabel("")
    _save(fig, F / "eda_length_distributions.png")
    df.groupby("label")[["n_chars", "n_tokens"]].describe().to_csv(paths.metrics / "length_stats.csv")

    # n-grams by sentiment
    for n, name, stop in ((1, "unigrams", True), (2, "bigrams", False), (3, "trigrams", False)):
        res = _top_ngrams(df, labels, n, 12, stop)
        fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.5))
        for ax, lab in zip(axes, labels):
            s = res[lab].sort_values()
            ax.barh(s.index, s.values, color=pal[lab]); ax.set_title(f"{lab}"); ax.tick_params(axis="y", labelsize=8)
        fig.suptitle(f"Top {name} by sentiment" + (" (stop-words removed)" if stop else ""))
        _save(fig, F / f"eda_top_{name}.png")
        pd.DataFrame({l: pd.Series(res[l].index) for l in labels}).to_csv(paths.metrics / f"top_{name}.csv", index=False)

    # hashtags / mentions
    tags = pd.Series([h.lower() for t in df.text for h in extract_entities(t)[1]]).value_counts().head(15)
    ment = pd.Series([m.lower() for t in df.text for m in extract_entities(t)[0]]).value_counts().head(15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, s, t in ((axes[0], tags, "Hashtags"), (axes[1], ment, "Mentions")):
        if len(s): s[::-1].plot(kind="barh", ax=ax, color="#6d597a")
        else: ax.text(0.5, 0.5, "none found", ha="center")
        ax.set_title(f"Most frequent {t}")
    _save(fig, F / "eda_hashtags_mentions.png")

    # optional supplementary word clouds
    try:
        from wordcloud import WordCloud
        fig, axes = plt.subplots(1, len(labels), figsize=(4.5 * len(labels), 3.6))
        for ax, lab in zip(axes, labels):
            txt = " ".join(df.loc[df.label == lab, "text"]).lower()
            ax.imshow(WordCloud(width=500, height=360, background_color="white", stopwords=set(ENGLISH_STOP_WORDS),
                                colormap="viridis").generate(txt)); ax.axis("off"); ax.set_title(lab)
        _save(fig, F / "eda_wordclouds.png")
    except ImportError:
        log("  wordcloud not installed - supplementary word clouds skipped")
    log(f"EDA figures written to {F}")
