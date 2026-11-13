"""Student Dropout Early-Warning System  -  run with:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Student Dropout Risk Prediction", page_icon=str(ROOT / "assets" / "icon.svg"), layout="wide")

from core import ui  # noqa: E402

ui.inject_css()
# keep what was typed on the Live Prediction form when the user visits another page (cleared only by "Reset form")
for _k in [k for k in st.session_state if str(k).startswith("in_")]:
    st.session_state[_k] = st.session_state[_k]
st.logo(str(ROOT / "assets" / "logo.svg"), icon_image=str(ROOT / "assets" / "icon.svg"), size="large")
nav = st.navigation([
    st.Page("views/home.py", title="Home", icon=":material/dashboard:", default=True),
    st.Page("views/shap_dashboard.py", title="SHAP Explanation", icon=":material/insights:"),
    st.Page("views/live_prediction.py", title="Live Prediction", icon=":material/bolt:"),
    st.Page("views/model_performance.py", title="Model Performance", icon=":material/analytics:"),
])
with st.sidebar:
    st.write("")
    st.toggle(":material/dark_mode: Dark mode", key="dark_mode")
nav.run()
