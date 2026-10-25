import json

import pandas as pd

from src.config import Paths
from src.report import build_report


def test_report_flags_failed_saved_model_verification(tmp_path):
    paths = Paths(mode="four", out=tmp_path).ensure()
    paths.metrics.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "distilbert",
        "val_macro_f1": 0.89,
        "test_accuracy": 0.93,
        "test_macro_f1": 0.92,
        "test_macro_f1_ci95_low": 0.90,
        "test_macro_f1_ci95_high": 0.94,
        "test_weighted_f1": 0.93,
        "train_seconds": 100,
        "inference_ms_per_1k_tweets": 5000,
        "model_size_mb": 268.5,
        "device_for_inference": "cuda",
        "all_checks_passed": False,
        "selected_by_val_macro_f1": True,
    }]).to_csv(paths.metrics / "leaderboard.csv", index=False)
    (paths.metrics / "verification.json").write_text(json.dumps({
        "checks": {"distilbert": {"test_macro_f1_match": False}}
    }))

    report = build_report(paths, "four")

    assert "Verification warning" in report
    assert "distilbert" in report
    assert "not used for selection" in report
