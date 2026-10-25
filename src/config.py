"""Central configuration. Paths are relative to the project root and can be
overridden with environment variables (no machine-specific absolute paths).

  TSP_DATA_DIR   folder holding twitter_training.csv / twitter_validation.csv  (default: ./data)
  TSP_OUT_DIR    folder for models/ and results/                                 (default: project root)
  TSP_LABEL_MODE four | three_exclude | three_merge                              (default: four)
  TSP_SEED       random seed                                                     (default: 42)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TSP_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("TSP_OUT_DIR", ROOT))
SEED = int(os.environ.get("TSP_SEED", 42))
DEFAULT_MODE = os.environ.get("TSP_LABEL_MODE", "four")

RAW_LABELS = ["Negative", "Neutral", "Positive", "Irrelevant"]
# four          : all four original labels (PRIMARY)
# three_exclude : Irrelevant rows are removed (Negative/Neutral/Positive only)
# three_merge   : Irrelevant is explicitly mapped to Neutral (documented alternative)
LABELS_BY_MODE = {
    "four": ["Negative", "Neutral", "Positive", "Irrelevant"],
    "three_exclude": ["Negative", "Neutral", "Positive"],
    "three_merge": ["Negative", "Neutral", "Positive"],
}


def labels_for_mode(mode: str) -> list[str]:
    if mode not in LABELS_BY_MODE:
        raise ValueError(f"Unknown label mode {mode!r}; choose from {list(LABELS_BY_MODE)}")
    return list(LABELS_BY_MODE[mode])


@dataclass(frozen=True)
class Paths:
    mode: str
    out: Path = OUT_DIR

    @property
    def models(self) -> Path: return self.out / "models" / self.mode
    @property
    def results(self) -> Path: return self.out / "results" / self.mode
    @property
    def metrics(self) -> Path: return self.results / "metrics"
    @property
    def figures(self) -> Path: return self.results / "figures"
    @property
    def reports(self) -> Path: return self.results / "reports"
    @property
    def topics(self) -> Path: return self.results / "topics"
    @property
    def splits(self) -> Path: return self.results / "splits"

    def ensure(self) -> "Paths":
        for p in (self.models, self.metrics, self.figures, self.reports, self.topics, self.splits):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(mode: str | None = None) -> Paths:
    mode = mode or DEFAULT_MODE
    labels_for_mode(mode)  # validate
    return Paths(mode=mode).ensure()
