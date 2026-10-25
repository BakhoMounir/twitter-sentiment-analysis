from pathlib import Path
import subprocess
import sys
import time
import importlib.util

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "health_check"
OUT.mkdir(parents=True, exist_ok=True)
results = []

def check(name, status, detail):
    results.append((name, status, str(detail)))
    print(f"[{status}] {name}: {detail}")

def run(name, cmd, timeout=900):
    print(f"\n--- {name} ---")
    start = time.time()
    try:
        p = subprocess.run(
            cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout
        )
        logfile = OUT / (name.lower().replace(" ", "_") + ".log")
        logfile.write_text(p.stdout, encoding="utf-8")
        status = "PASS" if p.returncode == 0 else "FAIL"
        check(name, status, f"exit={p.returncode}; log={logfile.relative_to(ROOT)}; {time.time()-start:.1f}s")
        if p.returncode:
            print("\n".join(p.stdout.splitlines()[-20:]))
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        check(name, "WARNING", f"Timed out after {timeout}s")
        return False

print("Project:", ROOT)
print("Python:", sys.version.split()[0])

# Required files
required = [
    "data/twitter_training.csv", "data/twitter_validation.csv",
    "src/config.py", "src/preprocessing.py", "src/data.py",
    "src/classical.py", "src/topics.py", "app.py",
    "scripts/verify_models.py",
]
missing = [f for f in required if not (ROOT / f).is_file()]
check("Required files", "FAIL" if missing else "PASS",
      ", ".join(missing) if missing else "All present")

# Saved models
try:
    from src.config import get_paths
    p = get_paths("four")
    artifacts = [
        p.models / "label_map.json",
        p.models / "classical" / "logreg.joblib",
        p.models / "classical" / "linear_svm.joblib",
        p.models / "classical" / "naive_bayes.joblib",
        p.models / "classical" / "random_forest.joblib",
        p.models / "distilbert" / "config.json",
    ]
    absent = [str(x.relative_to(ROOT)) for x in artifacts if not x.is_file()]
    check("Saved model artifacts", "FAIL" if absent else "PASS",
          ", ".join(absent) if absent else "All expected artifacts present")
except Exception as e:
    check("Project configuration", "FAIL", repr(e))

# Syntax and tests
run("Python syntax", [sys.executable, "-m", "compileall", "-q", "src", "scripts", "app.py"])
if importlib.util.find_spec("pytest"):
    run("Test suite", [sys.executable, "-m", "pytest", "-q"], timeout=900)
else:
    check("Test suite", "WARNING", "pytest is not installed in this environment")

# Recheck saved metrics and predictions
run("Model verification", [
    sys.executable, "scripts/verify_models.py",
    "--label-mode", "four", "--bootstrap", "200"
], timeout=1800)

# Output existence checks only; they do not prove output quality
for name, path in [
    ("Metrics outputs", ROOT / "results" / "four" / "metrics"),
    ("Topic outputs", ROOT / "results" / "four" / "topics"),
]:
    count = sum(1 for x in path.rglob("*") if x.is_file()) if path.exists() else 0
    check(name, "PASS" if count else "WARNING", f"{count} files found")

# Streamlit dependency
try:
    import streamlit
    check("Streamlit dependency", "PASS", streamlit.__version__)
except Exception:
    check("Streamlit dependency", "WARNING", "Streamlit unavailable; dashboard smoke test not run")

lines = [
    "# Twitter Sentiment Project — Health Check", "",
    f"- Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"- Python: {sys.version.split()[0]}",
    f"- Project: `{ROOT}`", "",
    "| Check | Status | Details |",
    "|---|---|---|",
]
for name, status, detail in results:
    lines.append(f"| {name} | **{status}** | {detail.replace('|', '/')} |")
lines += [
    "", "## Important limitations", "",
    "- Model verification checks metric reproducibility, not scientific validity.",
    "- Output-file presence does not establish topic quality or dashboard functionality.",
    "- Test-set similarity and generalization still require interpretation.",
    "- Dataset size and the four-class versus three-class specification must be documented.",
    "",
]
lines += [
    f"**Summary:** {sum(s == 'PASS' for _, s, _ in results)} PASS, "
    f"{sum(s == 'FAIL' for _, s, _ in results)} FAIL, "
    f"{sum(s == 'WARNING' for _, s, _ in results)} WARNING."
]
report = OUT / "health_check_report.md"
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nReport saved to:", report)
print(lines[-1])
