"""Headless run of every page (no browser):  python tests/test_app.py"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for page in ("views/home.py", "views/shap_dashboard.py", "views/live_prediction.py", "views/model_performance.py"):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    print(f"{page:30s} exceptions={len(at.exception)}")
    for e in at.exception:
        print("   ", e.value)
        print(e.stack_trace[-1] if e.stack_trace else "")
