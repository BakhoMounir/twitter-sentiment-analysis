"""Headless smoke test of the Streamlit app against already-generated artifacts (set TSP_OUT_DIR)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
assert not at.exception, at.exception
print("loaded; tabs:", len(at.tabs))
# single-text prediction
at.text_area[0].set_value("I love this game, it's awesome!"); at.button[0].click().run()
assert not at.exception, at.exception
print("prediction markdown:", [m.value for m in at.markdown if "->" in m.value][:1])
# empty input handled
at.text_area[0].set_value("   "); at.button[0].click().run()
assert not at.exception and any("enter some text" in w.value for w in at.warning), "empty-input warning missing"
print("empty input handled")
