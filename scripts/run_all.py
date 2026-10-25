#!/usr/bin/env python
"""Run pipeline stages.  Example:  python scripts/run_all.py --stages data eda classical topics report
Stages: data eda classical topics transformer report | all (= everything incl. transformer)."""
import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEFAULT_MODE, get_paths  # noqa: E402

DEFAULT = ["data", "eda", "classical", "topics", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=DEFAULT)
    ap.add_argument("--label-mode", default=DEFAULT_MODE, choices=["four", "three_exclude", "three_merge"])
    ap.add_argument("--fast", action="store_true", help="smaller grids / fewer trees (quick runs)")
    ap.add_argument("--skip-rf", action="store_true")
    ap.add_argument("--skip-preproc-compare", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=None, help="subsample train rows for the Transformer (CPU)")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="retrain DistilBERT even if a saved checkpoint exists")
    a = ap.parse_args()
    stages = ["data", "eda", "classical", "topics", "transformer", "report"] if "all" in a.stages else a.stages
    paths = get_paths(a.label_mode); mode = a.label_mode
    from src.evaluation import save_json
    timings = {}

    def run(name, fn):
        print(f"\n=== {name} ===", flush=True); t = time.time(); fn(); timings[name] = round(time.time() - t, 1)

    if "data" in stages:
        from src.data import prepare_splits
        run("data", lambda: print(json.dumps(prepare_splits(paths, mode)["splits"], indent=1)))
    if "eda" in stages:
        from src.eda import run_eda; run("eda", lambda: run_eda(paths, mode))
    if "classical" in stages:
        from src.classical import run_classical
        run("classical", lambda: print(run_classical(paths, mode, not a.skip_rf, not a.skip_preproc_compare, a.fast)
                                       .round(4).to_string()))
    if "topics" in stages:
        from src.topics import run_topics; run("topics", lambda: run_topics(paths, mode, a.fast))
    if "transformer" in stages:
        if (paths.models / "distilbert" / "config.json").exists() and not a.force:
            print("\n=== transformer === saved DistilBERT found -> skipping retraining (use --force to retrain; "
                  "use scripts/verify_models.py to re-check it)")
        else:
            from src.transformer import run_transformer
            run("transformer", lambda: run_transformer(paths, mode, epochs=a.epochs, max_train=a.max_train, max_len=a.max_len))
    envf = paths.metrics / "environment.json"
    env = json.loads(envf.read_text()) if envf.exists() else {}
    pk = {}
    for p in ("numpy", "pandas", "scikit-learn", "nltk", "torch", "transformers", "streamlit", "plotly"):
        try: pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError: pass
    import os
    env.update(python=platform.python_version(), platform=platform.platform(), cpu_count=os.cpu_count(), packages=pk,
               stage_seconds={**env.get("stage_seconds", {}), **timings}, label_mode=mode)
    save_json(env, envf)
    if "report" in stages:
        from src.report import build_report; run("report", lambda: build_report(paths, mode))
    print("\nDone. Outputs:", paths.results, "|", paths.models)


if __name__ == "__main__":
    main()