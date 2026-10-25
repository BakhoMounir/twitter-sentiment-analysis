"""Streamlit dashboard for sentiment classification and topic exploration.

Supports classical models, DistilBERT, and saved CNN/LSTM/Attention checkpoints
for both single-text and CSV batch prediction.

Run: streamlit run app.py
"""
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import OUT_DIR, Paths
from src import sentiment as S

st.set_page_config(page_title="Twitter Sentiment & Topics", layout="wide")

COLORS = {
    "Negative": "#d1495b",
    "Neutral": "#8d99ae",
    "Positive": "#2a9d8f",
    "Irrelevant": "#edae49",
}

DEEP_BASELINE_NAMES = {
    "cnn": "CNN",
    "lstm": "LSTM",
    "attention": "Attention",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_json(path: Path):
    """Read JSON if it exists; otherwise return None."""
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def read_csv(path_string: str, mtime: float):
    """Cache CSV reads and invalidate when the file modification time changes."""
    del mtime
    return pd.read_csv(path_string, keep_default_na=False)


def csv(path: Path):
    return read_csv(str(path), path.stat().st_mtime) if path.exists() else None


@st.cache_resource(show_spinner="Loading model...")
def get_classical(out: str, mode: str, name: str):
    return S.load_classical(Paths(mode, Path(out)), name)


@st.cache_resource(show_spinner="Loading deep-learning model...")
def get_deep_model(out: str, mode: str, name: str):
    """Load a saved CNN/LSTM/Attention checkpoint, including its training vocabulary."""
    import torch
    from src.deep_models import (
        LSTMSentimentClassifier,
        MultiKernelCNNClassifier,
        AttentionPoolingClassifier,
    )

    paths = Paths(mode, Path(out))
    checkpoint_path = paths.results / "deep_baselines" / f"{name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Deep-learning checkpoint not found: {checkpoint_path}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    vocab = checkpoint.get("vocab")
    checkpoint_labels = checkpoint.get("labels")
    max_length = checkpoint.get("max_length")

    if not isinstance(vocab, dict) or not checkpoint_labels or not max_length:
        raise ValueError(
            f"Checkpoint metadata is incomplete: {checkpoint_path}"
        )

    factories = {
        "lstm": LSTMSentimentClassifier,
        "cnn": MultiKernelCNNClassifier,
        "attention": AttentionPoolingClassifier,
    }

    model_name = checkpoint.get("model", name).lower()

    if model_name not in factories:
        raise ValueError(
            f"Unknown deep model type in checkpoint: {model_name}"
        )

    model = factories[model_name](
        len(vocab),
        len(checkpoint_labels),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "labels": list(checkpoint_labels),
        "max_length": int(max_length),
        "device": device,
    }


def predict_deep(
    bundle,
    texts: list[str],
    labels: list[str],
    batch_size: int = 64,
):
    """Run inference with the same token encoding used during training."""
    import numpy as np
    import torch

    token_re = re.compile(r"[\w#@']+", re.UNICODE)

    vocab = bundle["vocab"]
    max_length = bundle["max_length"]
    model = bundle["model"]
    device = bundle["device"]
    checkpoint_labels = bundle["labels"]

    encoded_ids = []
    encoded_lengths = []

    for text in texts:
        tokens = token_re.findall(str(text).lower())[:max_length]

        ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

        if not ids:
            ids = [vocab["<unk>"]]

        length = min(len(ids), max_length)
        ids = ids[:max_length]
        ids += [vocab["<pad>"]] * (max_length - len(ids))

        encoded_ids.append(ids)
        encoded_lengths.append(length)

    if not texts:
        return (
            [],
            np.empty((0, len(labels)), dtype=float),
            "softmax score (not calibrated)",
        )

    probs_chunks = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(encoded_ids), batch_size):
            batch_ids = encoded_ids[start:start + batch_size]
            batch_lengths = encoded_lengths[start:start + batch_size]

            ids_tensor = torch.tensor(
                batch_ids,
                dtype=torch.long,
                device=device,
            )

            lengths_tensor = torch.tensor(
                batch_lengths,
                dtype=torch.long,
                device=device,
            )

            logits = model(ids_tensor, lengths_tensor)
            probabilities = torch.softmax(logits, dim=1)

            probs_chunks.append(probabilities.cpu().numpy())

    raw_probs = np.concatenate(probs_chunks, axis=0)

    # Match checkpoint label order to the application's label order.
    missing = [
        label for label in labels
        if label not in checkpoint_labels
    ]

    if missing:
        raise ValueError(
            f"Labels missing from deep checkpoint: {missing}"
        )

    probs = raw_probs[
        :,
        [checkpoint_labels.index(label) for label in labels],
    ]

    predictions = [
        labels[index]
        for index in probs.argmax(axis=1)
    ]

    return predictions, probs, "softmax score (not calibrated)"


@st.cache_resource(show_spinner="Loading topic models...")
def get_topics(out: str, mode: str):
    import joblib

    path = Paths(mode, Path(out)).models / "topic_models.joblib"
    return joblib.load(path) if path.exists() else None


def safe_metric(value, digits=4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def find_deep_baseline_artifacts(path: Path):
    """Return artifact inventory for CNN/LSTM/Attention."""
    found = {}

    for key, display_name in DEEP_BASELINE_NAMES.items():
        model_file = path / f"{key}.pt"
        metrics_file = path / f"{key}_metrics.json"
        predictions_file = path / f"{key}_test_predictions.csv"
        history_file = path / f"{key}_history.csv"

        found[display_name] = {
            "model": model_file if model_file.exists() else None,
            "metrics": metrics_file if metrics_file.exists() else None,
            "predictions": (
                predictions_file if predictions_file.exists() else None
            ),
            "history": history_file if history_file.exists() else None,
        }

    return found


def show_deep_baseline_inventory(results_path: Path):
    """Display saved deep-learning experiment artifacts."""
    st.subheader("Additional deep-learning experiments")

    st.caption(
        "Inspect the saved CNN, LSTM, and Attention model weights, metrics, "
        "training histories, and held-out test predictions."
    )

    candidate_dirs = [
        results_path / "deep_baselines",
        results_path / "deep_baselines_fixed_smoke",
        results_path / "deep_baselines_smoke",
    ]

    existing_dirs = [
        path for path in candidate_dirs
        if path.exists()
    ]

    if not existing_dirs:
        st.info("No deep-baseline experiment directories were found.")
        return

    for folder in existing_dirs:
        with st.expander(
            f"Experiment artifacts: {folder.name}",
            expanded=(folder.name == "deep_baselines"),
        ):
            artifacts = find_deep_baseline_artifacts(folder)
            rows = []

            for model_name, files in artifacts.items():
                rows.append({
                    "Model": model_name,
                    "Weights": (
                        "Available" if files["model"] else "Missing"
                    ),
                    "Metrics": (
                        "Available" if files["metrics"] else "Missing"
                    ),
                    "Test predictions": (
                        "Available" if files["predictions"] else "Missing"
                    ),
                    "Training history": (
                        "Available" if files["history"] else "Missing"
                    ),
                })

            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
            )

            selected_model = st.selectbox(
                "Inspect experiment",
                list(artifacts),
                key=f"deep_artifact_select_{folder.name}",
            )

            selected = artifacts[selected_model]

            if selected["metrics"]:
                metrics = read_json(selected["metrics"])

                if metrics is not None:
                    st.markdown(
                        f"**{selected_model} metrics artifact**"
                    )
                    st.json(metrics)

            if selected["history"]:
                hist = csv(selected["history"])

                if hist is not None and not hist.empty:
                    st.markdown(
                        f"**{selected_model} training history**"
                    )

                    st.dataframe(
                        hist.tail(20),
                        hide_index=True,
                        width="stretch",
                    )

                    numeric_columns = [
                        col for col in hist.columns
                        if pd.api.types.is_numeric_dtype(
                            pd.to_numeric(hist[col], errors="coerce")
                        )
                    ]

                    if len(numeric_columns) >= 2:
                        x_col = numeric_columns[0]
                        y_candidates = numeric_columns[1:]

                        y_col = st.selectbox(
                            "History metric",
                            y_candidates,
                            key=(
                                f"deep_history_metric_{folder.name}_"
                                f"{selected_model}"
                            ),
                        )

                        plot_df = hist.copy()
                        plot_df[x_col] = pd.to_numeric(
                            plot_df[x_col],
                            errors="coerce",
                        )
                        plot_df[y_col] = pd.to_numeric(
                            plot_df[y_col],
                            errors="coerce",
                        )

                        plot_df = plot_df.dropna(
                            subset=[x_col, y_col]
                        )

                        if not plot_df.empty:
                            st.plotly_chart(
                                px.line(
                                    plot_df,
                                    x=x_col,
                                    y=y_col,
                                    title=(
                                        f"{selected_model}: "
                                        f"{y_col} over {x_col}"
                                    ),
                                ),
                                width="stretch",
                            )

            if selected["predictions"]:
                st.download_button(
                    f"Download {selected_model} experiment test predictions",
                    selected["predictions"].read_bytes(),
                    file_name=selected["predictions"].name,
                    mime="text/csv",
                    key=(
                        f"deep_predictions_{folder.name}_"
                        f"{selected_model}"
                    ),
                )# ---------------------------------------------------------------------------
# Sidebar and paths
# ---------------------------------------------------------------------------
results_root = OUT_DIR / "results"

modes = (
    sorted(
        d.name for d in results_root.glob("*")
        if (d / "metrics").exists()
    )
    if results_root.exists()
    else []
)

st.sidebar.title("Twitter Sentiment & Topics")

if not modes:
    st.error(
        "No results found. Place the CSV files in `data/` and run "
        "`python scripts/run_all.py` first (see README)."
    )
    st.stop()

mode = st.sidebar.selectbox(
    "Label mode",
    modes,
    index=modes.index("four") if "four" in modes else 0,
    help=(
        "four = primary 4-class task; "
        "three_* = documented 3-class alternatives"
    ),
)

P = Paths(mode, OUT_DIR)

labels = (
    S.get_labels(P)
    if (P.models / "label_map.json").exists()
    else None
)

models = S.available_models(P) if labels else {}

# Add deep-learning checkpoints to the prediction selector.
if labels:
    deep_dir = P.results / "deep_baselines"

    for deep_name in ("cnn", "lstm", "attention"):
        if (deep_dir / f"{deep_name}.pt").exists():
            models[deep_name] = "deep"

st.sidebar.caption(f"Artifacts: `{P.results}`")
st.sidebar.write(
    "Prediction models available:",
    ", ".join(models) or "none",
)

tabs = st.tabs([
    "Overview",
    "Sentiment analysis",
    "Topic exploration",
    "Trends & alerts",
    "Deep-learning experiments",
    "Reporting",
])

splits = {
    name: csv(P.splits / f"{name}.csv")
    for name in ("train", "val", "test")
}

available_splits = [
    frame for frame in splits.values()
    if frame is not None
]

labeled = (
    pd.concat(available_splits, ignore_index=True)
    if available_splits
    else None
)

topic_summary = read_json(P.topics / "topic_summary.json")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.title("Twitter Sentiment & Topic Analysis")

    st.caption(
        "Dataset-based sentiment classification, topic exploration, "
        "and model evaluation."
    )

    dv = read_json(P.metrics / "data_validation.json")

    if dv:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Training file rows",
            f"{dv['training_file']['rows_read']:,}",
        )

        c2.metric(
            "Validation file rows",
            f"{dv['validation_file']['rows_read']:,}",
        )

        c3.metric(
            "Train / val / test used",
            " / ".join(
                f"{dv['splits'][key]['rows']:,}"
                for key in ("train", "val", "test")
            ),
        )

        c4.metric(
            "Entities",
            dv["training_file"]["n_entities"],
        )

        st.caption(dv["splits"]["protocol"])

        with st.expander("Data validation details"):
            st.json({
                "training_file": dv["training_file"],
                "validation_file": dv["validation_file"],
                "cleaning": dv["cleaning"],
            })
    else:
        st.info("data_validation.json not found.")

    if (
        labeled is not None
        and labels
        and "label" in labeled.columns
    ):
        dist = (
            labeled["label"]
            .value_counts()
            .reindex(labels, fill_value=0)
            .rename_axis("label")
            .reset_index(name="count")
        )

        st.plotly_chart(
            px.bar(
                dist,
                x="label",
                y="count",
                color="label",
                color_discrete_map=COLORS,
                title="Class distribution (combined saved splits)",
            ),
            width="stretch",
        )

    st.subheader("Model performance")

    leaderboard = csv(P.metrics / "leaderboard.csv")
    comparison = csv(P.metrics / "model_comparison.csv")

    if leaderboard is not None:
        st.caption(
            "Validation metrics are used for model selection. "
            "Test metrics are reported separately. Consult the evaluation "
            "report for the exact split protocol and confidence-interval method."
        )

        st.dataframe(
            leaderboard.round(4),
            width="stretch",
            hide_index=True,
        )

        metric_columns = [
            column
            for column in (
                "test_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
            )
            if column in leaderboard.columns
        ]

        if metric_columns and "model" in leaderboard.columns:
            long = leaderboard.melt(
                id_vars="model",
                value_vars=metric_columns,
                var_name="metric",
                value_name="score",
            )

            st.plotly_chart(
                px.bar(
                    long,
                    x="model",
                    y="score",
                    color="metric",
                    barmode="group",
                    title="Held-out test metrics",
                ),
                width="stretch",
            )

        duplicates = csv(
            P.metrics / "near_duplicate_by_model.csv"
        )

        if (
            duplicates is not None
            and {
                "similarity_bin",
                "accuracy",
                "model",
            }.issubset(duplicates.columns)
        ):
            st.markdown(
                "**Test accuracy by similarity to the training set**"
            )

            st.caption(
                "Similarity is a diagnostic; high similarity alone "
                "does not prove data leakage."
            )

            st.plotly_chart(
                px.bar(
                    duplicates,
                    x="similarity_bin",
                    y="accuracy",
                    color="model",
                    barmode="group",
                    category_orders={
                        "similarity_bin": [
                            "<0.5 (novel)",
                            "0.5-0.7",
                            "0.7-0.9",
                            ">=0.9 (near-duplicate)",
                        ]
                    },
                ),
                width="stretch",
            )

        verification = read_json(P.metrics / "verification.json")

        if verification:
            st.caption(
                "Saved-model verification artifact is available; "
                "inspect verification.json for individual checks."
            )

            with st.expander("Verification details"):
                st.json(verification)

    elif comparison is not None:
        st.info(
            "Verified leaderboard not found. Showing the available "
            "model-comparison artifact instead."
        )

        st.dataframe(
            comparison.round(4),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No model comparison found. Run the relevant training "
            "and verification stages."
        )

    st.caption(
        "The preferred model should be selected using validation metrics, "
        "not test metrics."
    )

    figures = (
        sorted(P.figures.glob("eda_*.png"))
        if P.figures.exists()
        else []
    )

    if figures:
        st.subheader("Exploratory data analysis figures")

        selected_figures = st.multiselect(
            "Choose figures to display",
            [figure.name for figure in figures],
            default=[figure.name for figure in figures[:2]],
        )

        for figure_name in selected_figures:
            st.image(
                str(P.figures / figure_name),
                caption=figure_name,
                use_container_width=True,
            )

    if topic_summary:
        st.subheader("Topic-model summary")

        methods = topic_summary.get("methods", {})

        rows = [
            {
                "Method": method,
                "Number of topics/clusters": details.get("k"),
                "Selection rule": details.get("k_rule"),
            }
            for method, details in methods.items()
        ]

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
            )# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------
with tabs[1]:
    st.title("Sentiment analysis")

    if not models:
        st.warning(
            "No supported saved prediction models were discovered. "
            "Check the model artifacts and src/sentiment.py."
        )
    else:
        comparison = csv(P.metrics / "model_comparison.csv")
        best_models = []

        if (
            comparison is not None
            and {
                "selected_by_val_macro_f1",
                "model",
            }.issubset(comparison.columns)
        ):
            selected_mask = (
                comparison["selected_by_val_macro_f1"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            best_models = comparison.loc[
                selected_mask,
                "model",
            ].tolist()

        default_models = (
            [best_models[0]]
            if best_models and best_models[0] in models
            else [next(iter(models))]
        )

        choices = st.multiselect(
            "Prediction model(s)",
            list(models),
            default=default_models,
            help=(
                "Select one or more models supported by the "
                "current prediction API."
            ),
        )

        def run_models(texts):
            predictions = {}

            for model_name in choices:
                try:
                    kind = models[model_name]

                    if kind == "classical":
                        artifact = get_classical(
                            str(OUT_DIR),
                            mode,
                            model_name,
                        )

                        predictions[model_name] = S.predict_classical(
                            artifact,
                            texts,
                            labels,
                        )

                    elif kind == "deep":
                        bundle = get_deep_model(
                            str(OUT_DIR),
                            mode,
                            model_name,
                        )

                        predictions[model_name] = predict_deep(
                            bundle,
                            texts,
                            labels,
                        )

                    elif kind == "transformer":
                        predictions[model_name] = S.predict_transformer(
                            P,
                            texts,
                            labels,
                        )

                    else:
                        st.warning(
                            f"Skipping {model_name}: "
                            f"unsupported model type {kind!r}."
                        )

                except Exception as exc:
                    st.error(
                        f"{model_name} prediction failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            return predictions

        text = st.text_area(
            "Tweet text",
            height=100,
            placeholder="Type or paste a tweet...",
        )

        if st.button("Analyse", type="primary"):
            if not text.strip():
                st.warning("Please enter some text.")

            elif not choices:
                st.warning("Select at least one prediction model.")

            else:
                outputs = run_models([text])

                for model_name, (
                    predictions,
                    probabilities,
                    score_kind,
                ) in outputs.items():

                    st.markdown(
                        f"**{model_name}** → **{predictions[0]}** "
                        f"(top score {probabilities[0].max():.1%}; "
                        f"{score_kind})"
                    )

                    score_df = pd.DataFrame({
                        "label": labels,
                        "score": probabilities[0],
                    })

                    st.plotly_chart(
                        px.bar(
                            score_df,
                            x="label",
                            y="score",
                            color="label",
                            color_discrete_map=COLORS,
                            labels={
                                "label": "",
                                "score": "Score",
                            },
                            range_y=[0, 1],
                            height=260,
                        ),
                        width="stretch",
                    )

        st.divider()
        st.subheader("Batch CSV prediction")

        uploaded = st.file_uploader(
            "Upload a CSV containing a text column",
            type="csv",
        )

        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)

                if frame.empty or len(frame.columns) == 0:
                    st.warning(
                        "The uploaded CSV has no rows or columns."
                    )

                else:
                    default_index = (
                        list(frame.columns).index("text")
                        if "text" in frame.columns
                        else 0
                    )

                    text_column = st.selectbox(
                        "Text column",
                        frame.columns,
                        index=default_index,
                    )

                    if st.button("Predict batch") and choices:
                        texts = (
                            frame[text_column]
                            .fillna("")
                            .astype(str)
                            .tolist()
                        )

                        result = frame.copy()
                        outputs = run_models(texts)

                        for model_name, (
                            predictions,
                            probabilities,
                            _,
                        ) in outputs.items():

                            result[f"{model_name}_prediction"] = predictions

                            result[f"{model_name}_confidence"] = (
                                probabilities.max(axis=1).round(4)
                            )

                        st.session_state["batch_predictions"] = result

            except Exception as exc:
                st.error(
                    f"Could not process the uploaded file: "
                    f"{type(exc).__name__}: {exc}"
                )

        if "batch_predictions" in st.session_state:
            st.dataframe(
                st.session_state["batch_predictions"].head(200),
                width="stretch",
            )

            st.download_button(
                "Download predictions CSV",
                st.session_state["batch_predictions"]
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )# ---------------------------------------------------------------------------
# Topic exploration
# ---------------------------------------------------------------------------
with tabs[2]:
    st.title("Topic exploration")

    if not topic_summary:
        st.info(
            "Topic artifacts were not found. Run the topic-modeling stage."
        )

    else:
        methods = topic_summary.get("methods", {})

        if not methods:
            st.info("The topic summary contains no methods.")

        else:
            method = st.radio(
                "Method",
                list(methods),
                horizontal=True,
                format_func=str.upper,
            )

            method_data = methods[method]

            topic_labels = (
                read_json(P.topics / "topic_labels.json") or {}
            ).get(method, {})

            topics = method_data.get("topics", [])

            if not topics:
                st.info("No topic entries were found for this method.")

            else:
                topic_ids = [
                    topic["id"] for topic in topics
                ]

                topic_by_id = {
                    topic["id"]: topic
                    for topic in topics
                }

                topic_id = st.selectbox(
                    "Topic / cluster",
                    topic_ids,
                    format_func=lambda item: (
                        f"{item}: "
                        + (
                            topic_labels.get(str(item))
                            or ", ".join(
                                topic_by_id[item].get("terms", [])[:4]
                            )
                        )
                    ),
                )

                topic = topic_by_id[topic_id]
                label_text = (
                    topic_labels.get(str(topic_id))
                    or "Not reviewed"
                )

                st.markdown(
                    f"**Top terms:** "
                    f"{', '.join(topic.get('terms', []))}  \n"
                    f"**Size:** {topic.get('size', 0):,} tweets "
                    f"({topic.get('share', 0):.1%})  \n"
                    f"**Human label:** {label_text}"
                )

                st.markdown("**Representative texts**")

                for representative in topic.get(
                    "representative_tweets", []
                ):
                    st.write("- " + representative)

                sentiment_distribution = (
                    pd.Series(
                        topic.get("sentiment_distribution", {})
                    )
                    .rename_axis("label")
                    .reset_index(name="share")
                )

                if not sentiment_distribution.empty:
                    st.plotly_chart(
                        px.bar(
                            sentiment_distribution,
                            x="label",
                            y="share",
                            color="label",
                            color_discrete_map=COLORS,
                            title=(
                                "Sentiment distribution within this topic"
                            ),
                            range_y=[0, 1],
                        ),
                        width="stretch",
                    )

                pyldavis_file = P.topics / "lda_pyldavis.html"

                if pyldavis_file.exists():
                    st.download_button(
                        "Download interactive LDA view (open in browser)",
                        pyldavis_file.read_bytes(),
                        file_name=pyldavis_file.name,
                        mime="text/html",
                        key="pyldavis_download",
                    )

                st.caption(
                    f"Selection rule: "
                    f"{method_data.get('k_rule', 'Not documented')}. "
                    "Topics are exploratory structures, not validated ground truth."
                )

                topic_bundle = get_topics(str(OUT_DIR), mode)

                new_text = st.text_area(
                    "Assign topics to new text",
                    height=70,
                    key="topic_assignment_text",
                )

                if st.button("Assign topic"):
                    if not new_text.strip():
                        st.warning("Please enter some text.")

                    elif topic_bundle is None:
                        st.error("The topic-model bundle is missing.")

                    else:
                        try:
                            from src.topics import assign_topics

                            assignment = assign_topics(
                                topic_bundle,
                                [new_text],
                            ).iloc[0]

                            st.write(
                                f"LDA topic "
                                f"**{int(assignment.lda_topic)}** "
                                f"(confidence "
                                f"{float(assignment.lda_confidence):.0%}) | "
                                f"NMF topic "
                                f"**{int(assignment.nmf_topic)}** | "
                                f"KMeans cluster "
                                f"**{int(assignment.kmeans_cluster)}**"
                            )

                        except Exception as exc:
                            st.error(
                                f"Topic assignment failed: "
                                f"{type(exc).__name__}: {exc}"
                            )# ---------------------------------------------------------------------------
# Trends and alerts
# ---------------------------------------------------------------------------
with tabs[3]:
    st.title("Trends & alerts")

    st.caption(
        "The supplied dataset has no usable timestamps for chronological "
        "trend analysis. This page compares sentiment across entities and "
        "topics; it is not live monitoring or time-series spike detection."
    )

    if (
        labeled is None
        or not labels
        or "Negative" not in labels
        or "label" not in labeled.columns
    ):
        st.info(
            "Labeled splits or the Negative class are unavailable."
        )

    elif "entity" not in labeled.columns:
        st.info(
            "The saved splits do not contain an entity column."
        )

    else:
        col1, col2 = st.columns(2)

        threshold = col1.slider(
            "Negative-share alert threshold",
            0.05,
            0.90,
            0.40,
            0.05,
        )

        minimum_count = col2.number_input(
            "Minimum tweets per group",
            min_value=1,
            max_value=5000,
            value=50,
        )

        entity_summary = (
            labeled.assign(
                is_negative=labeled["label"].eq("Negative")
            )
            .groupby("entity")
            .agg(
                tweets=("is_negative", "size"),
                negative_share=("is_negative", "mean"),
            )
            .reset_index()
        )

        entity_summary = (
            entity_summary[
                entity_summary["tweets"] >= minimum_count
            ]
            .sort_values("negative_share", ascending=False)
        )

        entity_summary["alert"] = (
            entity_summary["negative_share"] >= threshold
        )

        st.metric(
            "Entities above threshold",
            int(entity_summary["alert"].sum()),
        )

        st.plotly_chart(
            px.bar(
                entity_summary,
                x="entity",
                y="negative_share",
                color="alert",
                hover_data=["tweets"],
                color_discrete_map={
                    True: COLORS["Negative"],
                    False: COLORS["Neutral"],
                },
                title="Negative sentiment share by entity",
            ).add_hline(
                y=threshold,
                line_dash="dash",
            ),
            width="stretch",
        )

        sentiment_mix = (
            pd.crosstab(
                labeled["entity"],
                labeled["label"],
                normalize="index",
            )
            .reindex(columns=labels, fill_value=0)
            .reset_index()
            .melt(
                "entity",
                var_name="label",
                value_name="share",
            )
        )

        st.plotly_chart(
            px.bar(
                sentiment_mix,
                x="entity",
                y="share",
                color="label",
                color_discrete_map=COLORS,
                title="Sentiment mix by entity",
            ),
            width="stretch",
        )

        if topic_summary and topic_summary.get("methods"):
            method = st.radio(
                "Topic-level view",
                list(topic_summary["methods"]),
                horizontal=True,
                format_func=str.upper,
                key="alert_topic_method",
            )

            topic_rows = [
                {
                    "topic": f"{method}-{topic.get('id')}",
                    "size": topic.get("size", 0),
                    "negative_share": (
                        topic.get(
                            "sentiment_distribution", {}
                        ).get("Negative", 0)
                    ),
                    "terms": ", ".join(
                        topic.get("terms", [])[:4]
                    ),
                }
                for topic in topic_summary["methods"][method].get(
                    "topics", []
                )
            ]

            topic_frame = pd.DataFrame(topic_rows)

            if not topic_frame.empty:
                topic_frame["alert"] = (
                    topic_frame["negative_share"] >= threshold
                )

                st.plotly_chart(
                    px.bar(
                        topic_frame,
                        x="topic",
                        y="negative_share",
                        color="alert",
                        hover_data=["terms", "size"],
                        color_discrete_map={
                            True: COLORS["Negative"],
                            False: COLORS["Neutral"],
                        },
                        title="Negative sentiment share by topic",
                    ).add_hline(
                        y=threshold,
                        line_dash="dash",
                    ),
                    width="stretch",
                )


# ---------------------------------------------------------------------------
# Deep-learning experiments
# ---------------------------------------------------------------------------
with tabs[4]:
    st.title("Deep-learning experiments")

    st.info(
        "CNN, LSTM, and Attention checkpoints are available in the "
        "Sentiment analysis tab for single-text and CSV predictions. "
        "This section lets you inspect their saved metrics, training "
        "histories, and held-out test predictions."
    )

    show_deep_baseline_inventory(P.results)


# ---------------------------------------------------------------------------
# Reporting and downloads
# ---------------------------------------------------------------------------
with tabs[5]:
    st.title("Reporting & downloads")

    download_items = [
        (
            "Verified leaderboard (CSV)",
            P.metrics / "leaderboard.csv",
            "text/csv",
        ),
        (
            "Per-class metrics (CSV)",
            P.metrics / "per_class_metrics.csv",
            "text/csv",
        ),
        (
            "Model comparison (CSV)",
            P.metrics / "model_comparison.csv",
            "text/csv",
        ),
        (
            "Final report (Markdown)",
            P.reports / "final_report.md",
            "text/markdown",
        ),
        (
            "Topic summary (JSON)",
            P.topics / "topic_summary.json",
            "application/json",
        ),
        (
            "Topic review sheet (CSV)",
            P.topics / "topic_review_sheet.csv",
            "text/csv",
        ),
        (
            "Entity sentiment table (CSV)",
            P.metrics / "entity_sentiment.csv",
            "text/csv",
        ),
        (
            "Topic assignments (CSV)",
            P.topics / "topic_assignments.csv",
            "text/csv",
        ),
    ]

    for label, path, mime in download_items:
        if path.exists():
            st.download_button(
                label,
                path.read_bytes(),
                file_name=path.name,
                mime=mime,
                key=f"download_{path.name}",
            )
        else:
            st.caption(
                f"{label}: unavailable (`{path.name}` not found)"
            )

    report_path = P.reports / "final_report.md"

    if report_path.exists():
        with st.expander("Preview final report"):
            st.markdown(
                report_path.read_text(encoding="utf-8")
            )