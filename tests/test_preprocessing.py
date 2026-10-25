from src.preprocessing import PreprocessConfig, clean_text, normalize_minimal, extract_entities

def test_negation_preserved():
    assert "not" in clean_text("I don't like it").split()
    assert "can not" in clean_text("I can't go")

def test_stopwords_keep_negation():
    out = clean_text("this is not a good game", PreprocessConfig(remove_stopwords=True)).split()
    assert "not" in out and "is" not in out

def test_urls_mentions_hashtags_repeats():
    out = clean_text("@Bob sooooo good http://t.co/x #Win")
    assert "http" not in out and "@user" in out and "win" in out and "sooo" not in out

def test_emoji_and_exclaim_kept():
    assert "😀" in clean_text("great 😀 !!") and "!" in clean_text("great !!")

def test_minimal_keeps_case_and_punct():
    assert normalize_minimal("Great &amp; FUN!! http://x.co") == "Great & FUN!! http"

def test_missing_text():
    assert clean_text(None) == "" and normalize_minimal(float("nan")) == ""

def test_entities():
    assert extract_entities("hi @a #b #c") == (["@a"], ["#b", "#c"])


def test_remove_mentions_and_entire_hashtags():
    text = "Great @Alice #NLP #SentimentAnalysis"
    cfg = PreprocessConfig(
        mention_mode="remove",
        hashtag_mode="remove",
    )
    out = clean_text(text, cfg)

    assert "alice" not in out
    assert "nlp" not in out
    assert "sentimentanalysis" not in out
    assert "great" in out


def test_strip_hashtags_preserves_words():
    out = clean_text(
        "#NLP #SentimentAnalysis",
        PreprocessConfig(hashtag_mode="strip"),
    )

    assert "nlp" in out
    assert "sentimentanalysis" in out
    assert "#" not in out


def test_keep_hashtags_preserves_hash_symbol():
    out = clean_text(
        "#NLP",
        PreprocessConfig(hashtag_mode="keep"),
    )

    assert "#nlp" in out


def test_invalid_hashtag_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        clean_text(
            "#NLP",
            PreprocessConfig(hashtag_mode="invalid"),
        )
