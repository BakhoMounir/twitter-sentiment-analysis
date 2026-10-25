"""Configurable tweet preprocessing.

* `normalize_minimal`  -> light normalisation for Transformers (keeps case, punctuation, emojis, negation)
* `clean_text` / `TextCleaner` -> configurable representation for classical models (sklearn-compatible)
Labels are never part of any input feature.
"""
from __future__ import annotations

import html
import re
import unicodedata
import warnings
from dataclasses import dataclass

from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOTICON_RE = re.compile(r"[:;=8xX][-o\^']?[)(DPpOo/\\|]+")
NEGATIONS = {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere",
             "cannot", "without", "hardly", "barely"}
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS      # negation is preserved
CONTRACTIONS = [(r"\bcan['’]t\b", "can not"), (r"\bwon['’]t\b", "will not"),
                (r"\bshan['’]t\b", "shall not"), (r"\bain['’]t\b", "is not"),
                (r"n['’]t\b", " not"), (r"['’]m\b", " am"), (r"['’]re\b", " are"),
                (r"['’]ve\b", " have"), (r"['’]ll\b", " will"), (r"['’]d\b", " would")]
_CONTR = [(re.compile(p, re.I), r) for p, r in CONTRACTIONS]
KEEP_PUNCT = {"!", "?"}


@dataclass
class PreprocessConfig:
    lowercase: bool = True
    url_token: str = ""                 # "" removes URLs
    mention_mode: str = "token"         # keep | token (-> @user) | remove
    hashtag_mode: str = "strip"         # keep | strip ('#' removed, word kept)
    expand_contractions: bool = True    # negation-preserving ("don't" -> "do not")
    max_repeat: int = 2                 # "soooo" -> "soo"; 0 disables
    remove_punct: bool = True           # drops punctuation tokens except ! ? (and emoji if keep_emoji)
    keep_emoji: bool = True
    remove_stopwords: bool = False      # negations never removed
    lemmatize: bool = False             # POS-aware WordNet lemmatisation (needs NLTK data; falls back)
    stem: bool = False                  # alternative to lemmatize, evaluated separately


def normalize_minimal(text) -> str:
    """Minimal normalisation for Transformer input."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = unicodedata.normalize("NFKC", html.unescape(str(text)))
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub("http", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_entities(text) -> tuple[list[str], list[str]]:
    t = "" if text is None else str(text)
    return MENTION_RE.findall(t), HASHTAG_RE.findall(t)


_TOK = TweetTokenizer(preserve_case=True, reduce_len=False, strip_handles=False)
_STEM = PorterStemmer()
_LEMMA = {"obj": None, "ok": None, "tagger_ok": None}


def _lemmatizer():
    if _LEMMA["ok"] is None:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            try:
                lem.lemmatize("tests")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                lem.lemmatize("tests")
            _LEMMA.update(obj=lem, ok=True)
            try:
                nltk.pos_tag(["a", "test"])
                _LEMMA["tagger_ok"] = True
            except LookupError:
                for r in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
                    nltk.download(r, quiet=True)
                try:
                    nltk.pos_tag(["a", "test"]); _LEMMA["tagger_ok"] = True
                except Exception:
                    _LEMMA["tagger_ok"] = False
        except Exception as e:  # pragma: no cover
            warnings.warn(f"WordNet unavailable ({e}); lemmatisation disabled.")
            _LEMMA["ok"] = False
    return _LEMMA["obj"] if _LEMMA["ok"] else None


def lemmatizer_available() -> bool:
    return _lemmatizer() is not None


def _lemmatize(tokens: list[str]) -> list[str]:
    lem = _lemmatizer()
    if lem is None:
        return tokens
    if _LEMMA["tagger_ok"]:
        import nltk
        pos = [{"J": "a", "V": "v", "R": "r"}.get(tag[0], "n") for _, tag in nltk.pos_tag(tokens)]
    else:
        pos = ["n"] * len(tokens)
    return [lem.lemmatize(t, p) if t.isalpha() else t for t, p in zip(tokens, pos)]


def clean_text(text, cfg: PreprocessConfig | None = None) -> str:
    cfg = cfg or PreprocessConfig()
    if text is None or (isinstance(text, float) and text != text):
        return ""
    t = unicodedata.normalize("NFKC", html.unescape(str(text)))
    t = TAG_RE.sub(" ", t)
    t = URL_RE.sub(f" {cfg.url_token} " if cfg.url_token else " ", t)
    if cfg.mention_mode == "token":
        t = MENTION_RE.sub("@user", t)
    elif cfg.mention_mode == "remove":
        t = MENTION_RE.sub(" ", t)
    elif cfg.mention_mode != "keep":
        raise ValueError(
            "mention_mode must be 'keep', 'token', or 'remove'"
        )

    if cfg.hashtag_mode == "remove":
        t = HASHTAG_RE.sub(" ", t)
    elif cfg.hashtag_mode == "strip":
        t = HASHTAG_RE.sub(lambda m: " " + m.group(0)[1:] + " ", t)
    elif cfg.hashtag_mode != "keep":
        raise ValueError(
            "hashtag_mode must be 'keep', 'strip', or 'remove'"
        )
    if cfg.expand_contractions:
        for rx, rep in _CONTR:
            t = rx.sub(rep, t)
    if cfg.max_repeat > 0:
        t = re.sub(r"(.)\1{%d,}" % cfg.max_repeat, lambda m: m.group(1) * cfg.max_repeat, t)
    if cfg.lowercase:
        t = t.lower()
    toks = _TOK.tokenize(t)
    if cfg.remove_punct:
        keep = []
        for tok in toks:
            if re.search(r"\w", tok) or tok in KEEP_PUNCT:
                keep.append(tok)
            elif cfg.keep_emoji and (any(ord(c) >= 0x2190 for c in tok) or EMOTICON_RE.fullmatch(tok)):
                keep.append(tok)
        toks = keep
    if cfg.remove_stopwords:
        toks = [x for x in toks if x.lower() not in STOPWORDS]
    if cfg.lemmatize:
        toks = _lemmatize(toks)
    elif cfg.stem:
        toks = [_STEM.stem(x) if x.isalpha() else x for x in toks]
    return " ".join(toks)


class TextCleaner(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper so cleaning travels inside a saved Pipeline."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cfg = self.config or PreprocessConfig()
        return [clean_text(t, cfg) for t in X]
