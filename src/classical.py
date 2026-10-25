"""Classical supervised models: TF-IDF + {LogReg, LinearSVM, RandomForest, ComplementNB}.

Protocol: fit on `train`, tune/select on `val`, report `test` (official validation file) once.
Vectorisers are fitted on training text only.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import SEED, Paths, labels_for_mode
from .data import load_splits
from .evaluation import evaluate, plot_confusion, save_json, upsert_comparison
from .preprocessing import PreprocessConfig, TextCleaner, clean_text, lemmatizer_available

TFIDF = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=300_000)


def _clean(texts, cfg):
    return [clean_text(t, cfg) for t in texts]


def compare_preprocessing(sp, labels, log=print) -> tuple[pd.DataFrame, PreprocessConfig]:
    """Evaluate alternative cleaning recipes with one fixed model (LogReg) on VAL only."""
    base = PreprocessConfig()
    variants = {
        "light (no contraction/repeat handling, keep #/@)": replace(
            base, mention_mode="keep", hashtag_mode="keep", expand_contractions=False,
            max_repeat=0, remove_punct=False),
        "default (negation-preserving)": base,
        "default + stopword removal": replace(base, remove_stopwords=True),
        "default + stemming": replace(base, stem=True),
    }
    if lemmatizer_available():
        variants["default + POS-aware lemmatisation"] = replace(base, lemmatize=True)
    else:
        log("  [skip] lemmatisation variant: NLTK WordNet data unavailable")
    rows = []
    for name, cfg in variants.items():
        t0 = time.time()
        Xtr, Xva = _clean(sp["train"].text, cfg), _clean(sp["val"].text, cfg)
        vec = TfidfVectorizer(**TFIDF)
        A = vec.fit_transform(Xtr)
        clf = LogisticRegression(C=4, max_iter=1000).fit(A, sp["train"].label)
        m = evaluate(sp["val"].label, clf.predict(vec.transform(Xva)), labels)
        rows.append({"preprocessing": name, "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                     "n_features": A.shape[1], "seconds": round(time.time() - t0, 1), "_cfg": cfg})
        log(f"  {name:55s} val macro-F1={m['macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    best_i = df["val_macro_f1"].idxmax()
    default_i = df.index[df["preprocessing"].str.startswith("default (")][0]
    if df.loc[best_i, "val_macro_f1"] - df.loc[default_i, "val_macro_f1"] < 0.002:   # ties -> simplest recipe
        best_i = default_i
    best_cfg = df.loc[best_i, "_cfg"]
    return df.drop(columns="_cfg"), best_cfg


def _max_cos(Q, R, chunk=250):
    out = []
    for i in range(0, Q.shape[0], chunk):
        out.append((Q[i:i + chunk] @ R.T).max(axis=1).toarray().ravel())
    return np.concatenate(out)


def near_duplicate_diagnostic(paths, Xtr, Xva, Xte, yte, test_pred, finals, log=print, thr=0.9):
    """How much of val/test is a near-copy of a training tweet (cosine >= thr)? Explains inflated scores."""
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.2, sublinear_tf=True).fit(Xtr)
    R = v.transform(Xtr)
    sim_te = _max_cos(v.transform(Xte), R)
    rng = np.random.RandomState(SEED); idx = rng.choice(len(Xva), min(2000, len(Xva)), replace=False)
    sim_va = _max_cos(v.transform([Xva[i] for i in idx]), R)
    best = max(finals, key=lambda f: finals[f][0][0])
    ok = (test_pred[best] == yte); hi = sim_te >= thr
    res = {"threshold_cosine": thr, "model_used": best,
           "test": {"n": int(len(yte)), "frac_near_duplicate_of_train": float(hi.mean()),
                    "accuracy_on_near_duplicates": float(ok[hi].mean()) if hi.any() else None,
                    "accuracy_on_novel": float(ok[~hi].mean()) if (~hi).any() else None,
                    "median_max_similarity": float(np.median(sim_te))},
           "val_sample": {"n": int(len(idx)), "frac_near_duplicate_of_train": float((sim_va >= thr).mean()),
                          "median_max_similarity": float(np.median(sim_va))}}
    save_json(res, paths.metrics / "near_duplicate_diagnostic.json")
    log(f"  near-duplicate check: {res['test']['frac_near_duplicate_of_train']:.1%} of test tweets have a train tweet with cosine>={thr}; "
        f"acc on those={res['test']['accuracy_on_near_duplicates']}, on novel={res['test']['accuracy_on_novel']}")


def run_classical(paths: Paths, mode: str, include_rf: bool = True, compare_prep: bool = True,
                  fast: bool = False, log=print) -> pd.DataFrame:
    labels = labels_for_mode(mode)
    sp = load_splits(paths)
    ytr, yva, yte = sp["train"].label, sp["val"].label, sp["test"].label

    # 1) preprocessing choice (validation only)
    if compare_prep:
        log("Comparing preprocessing recipes (LogReg, validation macro-F1)")
        prep_df, cfg = compare_preprocessing(sp, labels, log)
        prep_df.to_csv(paths.metrics / "preprocessing_comparison.csv", index=False)
    else:
        cfg = PreprocessConfig()
    log(f"Using preprocessing: {asdict(cfg)}")
    cleaner = TextCleaner(cfg)
    Xtr, Xva, Xte = (_clean(sp[k].text, cfg) for k in ("train", "val", "test"))

    # 2) shared TF-IDF (fit on train only)
    vec = TfidfVectorizer(**TFIDF)
    Atr = vec.fit_transform(Xtr); Ava = vec.transform(Xva); Ate = vec.transform(Xte)
    log(f"TF-IDF features: {Atr.shape[1]:,}")

    families: dict[str, list] = {
        "logreg": [({"C": C, "class_weight": cw}, lambda C=C, cw=cw: LogisticRegression(C=C, class_weight=cw, max_iter=1000))
                   for cw in (None, "balanced") for C in ((4,) if fast else (1, 4, 10))],
        "linear_svm": [({"C": C, "class_weight": cw}, lambda C=C, cw=cw: LinearSVC(C=C, class_weight=cw, random_state=SEED))
                       for cw in (None, "balanced") for C in ((0.5,) if fast else (0.1, 0.5, 1.0))],
        "naive_bayes": [({"alpha": a}, lambda a=a: ComplementNB(alpha=a)) for a in ((0.5,) if fast else (0.1, 0.5, 1.0))],
    }
    search_rows, finals, test_pred = [], {}, {}
    for fam, grid in families.items():
        best = None
        for params, make in grid:
            clf = make(); t0 = time.time(); clf.fit(Atr, ytr); dt = time.time() - t0
            f1 = evaluate(yva, clf.predict(Ava), labels)["macro_f1"]
            search_rows.append({"family": fam, "params": str(params), "val_macro_f1": f1, "fit_seconds": round(dt, 1)})
            if best is None or f1 > best[0]:
                best = (f1, params, clf, dt)
        finals[fam] = (best, vec, Atr.shape[1])
        log(f"  {fam}: best val macro-F1={best[0]:.4f} {best[1]}")

    if include_rf:
        rvec = TfidfVectorizer(**{**TFIDF, "max_features": 20_000, "min_df": 3})
        Rtr = rvec.fit_transform(Xtr)
        n_est = 100 if fast else 250
        clf = RandomForestClassifier(n_estimators=n_est, max_features="sqrt", n_jobs=-1, random_state=SEED)
        t0 = time.time(); clf.fit(Rtr, ytr); dt = time.time() - t0
        f1 = evaluate(yva, clf.predict(rvec.transform(Xva)), labels)["macro_f1"]
        search_rows.append({"family": "random_forest", "params": str({"n_estimators": n_est, "max_features": "sqrt",
                            "tfidf_max_features": 20000}), "val_macro_f1": f1, "fit_seconds": round(dt, 1)})
        finals["random_forest"] = ((f1, {"n_estimators": n_est}, clf, dt), rvec, Rtr.shape[1])
        log(f"  random_forest: val macro-F1={f1:.4f} ({dt:.0f}s)")
    pd.DataFrame(search_rows).to_csv(paths.metrics / "hyperparameter_search.csv", index=False)

    # 3) final evaluation on the held-out official validation file (one pass, no tuning)
    (paths.models / "classical").mkdir(parents=True, exist_ok=True)
    for fam, ((val_f1, params, clf, fit_s), v, nfeat) in finals.items():
        pipe = Pipeline([("clean", cleaner), ("tfidf", v), ("clf", clf)])   # already-fitted parts
        t0 = time.time(); pred = pipe.predict(sp["test"].text); inf = time.time() - t0
        test_pred[fam] = np.asarray(pred)
        m = evaluate(yte, pred, labels)
        m.update(model=fam, params=params, val_macro_f1=val_f1, fit_seconds=fit_s)
        m["inference_ms_per_1k"] = inf / max(len(yte), 1) * 1e6
        save_json(m, paths.metrics / f"classical_{fam}.json")
        plot_confusion(m["confusion_matrix"], labels, f"{fam} - test (official validation file)",
                       paths.figures / f"confusion_{fam}.png")
        joblib.dump(pipe, paths.models / "classical" / f"{fam}.joblib", compress=3)
        upsert_comparison(paths, {
            "model": fam, "family": "classical", "params": str(params), "n_features": nfeat,
            "val_macro_f1": val_f1, "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
            "test_weighted_f1": m["weighted_f1"], "train_seconds": round(fit_s, 1),
            "inference_ms_per_1k": round(m["inference_ms_per_1k"], 1)})
        log(f"  TEST {fam:14s} acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")

    near_duplicate_diagnostic(paths, Xtr, Xva, Xte, np.asarray(yte), test_pred, finals, log)

    # interpretable top terms of the logistic regression
    lr = finals["logreg"][0][2]
    names = np.array(vec.get_feature_names_out())
    top = {c: names[np.argsort(lr.coef_[i])[-15:][::-1]].tolist() for i, c in enumerate(lr.classes_)}
    save_json(top, paths.metrics / "logreg_top_terms.json")
    save_json({"labels": labels, "mode": mode, "preprocessing": asdict(cfg), "tfidf": {**TFIDF, "ngram_range": [1, 2]}},
              paths.models / "label_map.json")
    return pd.read_csv(paths.metrics / "model_comparison.csv")
