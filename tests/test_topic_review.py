from scripts.apply_topic_review import clean_terms


def test_removes_common_artifacts_and_duplicates():
    assert clean_terms("Twitter, pic, unk, great great, https://t.co/abc, youtube") == ["great"]


def test_handles_missing_values():
    assert clean_terms(None) == []
