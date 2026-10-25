"""Topic discovery: LDA (counts), NMF (TF-IDF) and KMeans (TF-IDF), fitted on TRAIN text only.

Topics are descriptive structure, not validated ground truth. Number of topics is chosen with a documented
rule (UMass coherence for LDA/NMF, silhouette reported for KMeans) and may be overridden with
TSP_LDA_K / TSP_NMF_K / TSP_KMEANS_K. Human labels are supplied via results/<mode>/topics/topic_labels.json.
"""
from __future__ import annotations

import html
import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

from .config import SEED, Paths, labels_for_mode
from .data import load_splits
from .evaluation import save_json

TOKEN = r"(?u)\b[a-z][a-z]{2,}\b"
EXTRA_STOP = {
    "just", "like", "get", "got", "really", "going", "know", "want", "one", "think", "make", "good", "new",
    # Frequent dataset / URL artefacts that are not useful topic descriptors.
    "pic", "twitter", "com", "youtu", "youtube", "http", "https", "unk", "amp",
}
NMF_MAX_ITER = 800
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def topic_clean(t) -> str:
    t = html.unescape(str(t)).lower()
    t = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", t)
    return re.sub(r"[^a-z\s]", " ", t)


def umass_coherence(B: sparse.csr_matrix, top_idx: np.ndarray) -> float:
    """Mean UMass coherence of topics. B = binary doc-term matrix, top_idx = (k, N) ranked word ids."""
    B = B.tocsc(); scores = []
    for words in top_idx:
        sub = B[:, words]; co = (sub.T @ sub).toarray(); d = np.diag(co)
        s = sum(np.log((co[m, l] + 1.0) / max(d[l], 1.0)) for m in range(1, len(words)) for l in range(m))
        scores.append(s / (len(words) * (len(words) - 1) / 2))
    return float(np.mean(scores))


def _top_terms(components, names, n=10):
    idx = np.argsort(components, axis=1)[:, ::-1][:, :n]
    return idx, [[names[i] for i in row] for row in idx]


def _k(env, default):
    return int(os.environ[env]) if env in os.environ else default


def _rep(scores: np.ndarray, texts: np.ndarray, n=3):
    order = np.argsort(scores)[::-1]
    out = []
    for i in order:
        if len(texts[i].split()) >= 4 and texts[i] not in out:
            out.append(texts[i])
        if len(out) == n: break
    return out


def run_topics(paths: Paths, mode: str, fast: bool = False, log=print) -> dict:
    labels = labels_for_mode(mode)
    sp = load_splits(paths)
    tr = sp["train"]; rng = np.random.RandomState(SEED)
    docs_tr = np.array([topic_clean(t) for t in tr.text]); raw_tr = tr.text.values
    summary: dict = {"label_mode": mode, "fit_on": "train split only", "methods": {}}

    # ---------------- LDA ----------------
    cv = CountVectorizer(stop_words=STOP, token_pattern=TOKEN, min_df=5, max_df=0.5, max_features=20_000)
    Ctr = cv.fit_transform(docs_tr); cnames = cv.get_feature_names_out()
    B = (Ctr > 0).astype(np.float32).tocsr()
    sel_idx = rng.choice(Ctr.shape[0], min(15_000 if not fast else 6_000, Ctr.shape[0]), replace=False)
    hold = cv.transform([topic_clean(t) for t in sp["val"].text.values[:3000]])
    grid = [5, 8, 10] if fast else [5, 8, 10, 12, 15, 20]
    sel = []
    for k in grid:
        t0 = time.time()
        m = LatentDirichletAllocation(k, learning_method="online", max_iter=8, batch_size=2048, random_state=SEED).fit(Ctr[sel_idx])
        ti, _ = _top_terms(m.components_, cnames)
        sel.append({"k": k, "umass_coherence": umass_coherence(B[sel_idx], ti),
                    "heldout_perplexity": float(m.perplexity(hold)), "seconds": round(time.time() - t0, 1)})
        log(f"  LDA k={k:2d} coherence={sel[-1]['umass_coherence']:.3f} perplexity={sel[-1]['heldout_perplexity']:.0f}")
    pd.DataFrame(sel).to_csv(paths.metrics / "lda_k_selection.csv", index=False)
    # The project specification requires the final LDA model to use 15–20 topics.
    # Compare the broader grid above, but select only among compliant candidates
    # unless an explicit environment override is provided (and validated).
    default_lda = max((r for r in sel if 15 <= int(r["k"]) <= 20),
                      key=lambda r: r["umass_coherence"])["k"]
    k_lda = _k("TSP_LDA_K", int(default_lda))
    if not 15 <= k_lda <= 20:
        raise ValueError(f"TSP_LDA_K must be between 15 and 20 for the required final model; got {k_lda}")
    lda = LatentDirichletAllocation(k_lda, learning_method="online", max_iter=12 if not fast else 6, batch_size=2048,
                                    random_state=SEED).fit(Ctr)
    lda_tr = lda.transform(Ctr)
    _, lda_terms = _top_terms(lda.components_, cnames)
    lda_ti, _ = _top_terms(lda.components_, cnames)
    lda_coh = umass_coherence(B, lda_ti)
    summary["methods"]["lda"] = {"k": k_lda, "k_rule": "max mean UMass coherence over the grid (override: TSP_LDA_K)",
                                 "umass_coherence_train": lda_coh, "terms": lda_terms}

    # ---------------- TF-IDF (NMF + KMeans) ----------------
    tv = TfidfVectorizer(stop_words=STOP, token_pattern=TOKEN, min_df=5, max_df=0.5, max_features=20_000, sublinear_tf=True)
    Ttr = tv.fit_transform(docs_tr); tnames = tv.get_feature_names_out(); TB = (Ttr > 0).astype(np.float32).tocsr()
    nsel = []
    for k in ([8, 10] if fast else [5, 8, 10, 12, 15]):
        m = NMF(k, init="nndsvd", max_iter=NMF_MAX_ITER, tol=1e-3, random_state=SEED).fit(Ttr)
        ti, _ = _top_terms(m.components_, tnames)
        nsel.append({"k": k, "umass_coherence": umass_coherence(TB, ti), "reconstruction_err": float(m.reconstruction_err_)})
        log(f"  NMF k={k:2d} coherence={nsel[-1]['umass_coherence']:.3f}")
    pd.DataFrame(nsel).to_csv(paths.metrics / "nmf_k_selection.csv", index=False)
    k_nmf = _k("TSP_NMF_K", int(max(nsel, key=lambda r: r["umass_coherence"])["k"]))
    nmf = NMF(k_nmf, init="nndsvd", max_iter=NMF_MAX_ITER, tol=1e-3, random_state=SEED).fit(Ttr)
    nmf_tr = nmf.transform(Ttr)
    _, nmf_terms = _top_terms(nmf.components_, tnames)
    summary["methods"]["nmf"] = {"k": k_nmf, "k_rule": "max mean UMass coherence over the grid (override: TSP_NMF_K)",
                                 "umass_coherence_train": umass_coherence(TB, _top_terms(nmf.components_, tnames)[0]),
                                 "terms": nmf_terms}

    # ---------------- KMeans ----------------
    sil_idx = rng.choice(Ttr.shape[0], min(4000, Ttr.shape[0]), replace=False); ksel = []
    for k in ([8, 10] if fast else [6, 8, 10, 12, 15]):
        km_ = MiniBatchKMeans(k, n_init=3, random_state=SEED, batch_size=2048).fit(Ttr)
        ksel.append({"k": k, "silhouette_cosine_sample": float(silhouette_score(Ttr[sil_idx], km_.labels_[sil_idx], metric="cosine")),
                     "inertia": float(km_.inertia_)})
    pd.DataFrame(ksel).to_csv(paths.metrics / "kmeans_k_selection.csv", index=False)
    k_km = _k("TSP_KMEANS_K", 10)
    km = MiniBatchKMeans(k_km, n_init=5, random_state=SEED, batch_size=2048).fit(Ttr)
    km_tr = km.labels_
    km_terms = [[tnames[i] for i in np.argsort(c)[::-1][:10]] for c in km.cluster_centers_]
    summary["methods"]["kmeans"] = {"k": k_km, "k_rule": "documented default 10 (override: TSP_KMEANS_K); silhouette reported, "
                                    "low silhouette is typical for short texts and is NOT proof of semantic quality",
                                    "silhouette_cosine_sample": next((r["silhouette_cosine_sample"] for r in ksel if r["k"] == k_km), None),
                                    "terms": km_terms}

    # ---------------- assignments for every split + per-topic summaries ----------------
    bundle = {"lda_vec": cv, "lda": lda, "tfidf_vec": tv, "nmf": nmf, "kmeans": km, "labels": labels}
    joblib.dump(bundle, paths.models / "topic_models.joblib", compress=3)
    frames = []
    for name in ("train", "val", "test"):
        d = sp[name].copy(); cl = [topic_clean(t) for t in d.text]
        Pl = lda.transform(cv.transform(cl)); T = tv.transform(cl)
        d["split"] = name
        d["lda_topic"], d["lda_conf"] = Pl.argmax(1), Pl.max(1).round(4)
        d["nmf_topic"] = nmf.transform(T).argmax(1)
        d["kmeans_cluster"] = km.predict(T)
        frames.append(d)
    A = pd.concat(frames, ignore_index=True)
    A.to_csv(paths.topics / "topic_assignments.csv", index=False)

    scores = {"lda": lda_tr, "nmf": nmf_tr}
    colmap = {"lda": "lda_topic", "nmf": "nmf_topic", "kmeans": "kmeans_cluster"}
    # Training assignments are used for representative examples, keeping each
    # example within the topic/cluster it is intended to describe.
    train_assignments = {
        "lda": lda_tr.argmax(axis=1),
        "nmf": nmf_tr.argmax(axis=1),
        "kmeans": km.labels_,
    }
    review = []
    for meth in ("lda", "nmf", "kmeans"):
        topics = []
        for t in range(summary["methods"][meth]["k"]):
            sub = A[A[colmap[meth]] == t]
            member_idx = np.flatnonzero(train_assignments[meth] == t)
            if meth == "kmeans":
                dist = km.transform(Ttr)[:, t]
                rep = _rep(-dist[member_idx], raw_tr[member_idx])
            else:
                rep = _rep(scores[meth][member_idx, t], raw_tr[member_idx])
            sd = sub["label"].value_counts(normalize=True).reindex(labels, fill_value=0).round(4).to_dict()
            topics.append({"id": t, "terms": summary["methods"][meth]["terms"][t], "size": int(len(sub)),
                           "share": round(len(sub) / len(A), 4), "representative_tweets": rep,
                           "sentiment_distribution": sd})
            review.append({"method": meth, "topic_id": t, "top_terms": ", ".join(summary["methods"][meth]["terms"][t][:8]),
                           **{f"example_{i+1}": (rep[i] if i < len(rep) else "") for i in range(3)},
                           "proposed_label": "", "reviewer": "", "notes": ""})
        summary["methods"][meth]["topics"] = topics
    save_json(summary, paths.topics / "topic_summary.json")
    pd.DataFrame(review).to_csv(paths.topics / "topic_review_sheet.csv", index=False)
    lab = paths.topics / "topic_labels.json"
    if not lab.exists():
        lab.write_text(json.dumps({m: {str(t): None for t in range(summary["methods"][m]["k"])} for m in summary["methods"]}, indent=2))

    try:  # optional interactive LDA view
        import pyLDAvis
        try:
            import pyLDAvis.lda_model as pv
        except ImportError:
            import pyLDAvis.sklearn as pv
        from pyLDAvis._prepare import js_PCoA
        real_mds = lambda d: np.real(js_PCoA(d))
        pyLDAvis.save_html(pv.prepare(lda, Ctr, cv, mds=real_mds), str(paths.topics / "lda_pyldavis.html"))
        log("  pyLDAvis HTML saved")
    except Exception as e:
        log(f"  pyLDAvis skipped ({type(e).__name__}: {str(e)[:120]}) - optional, other outputs unaffected")
    log(f"Topics: LDA k={k_lda}, NMF k={k_nmf}, KMeans k={k_km}")
    return summary


def assign_topics(bundle: dict, texts: list[str]) -> pd.DataFrame:
    cl = [topic_clean(t) for t in texts]
    P = bundle["lda"].transform(bundle["lda_vec"].transform(cl)); T = bundle["tfidf_vec"].transform(cl)
    return pd.DataFrame({"lda_topic": P.argmax(1), "lda_confidence": P.max(1),
                         "nmf_topic": bundle["nmf"].transform(T).argmax(1),
                         "kmeans_cluster": bundle["kmeans"].predict(T)})
