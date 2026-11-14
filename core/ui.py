"""Shared Streamlit pieces: cached loaders, styling, charts and the follow-up block."""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import config as cfg
from . import data_prep as dp
from . import explain as ex
from . import notify
from .predictor import Predictor, Result

_orig_plotly = st.plotly_chart


def _plotly_no_toolbar(fig, *a, **kw):
    kw.setdefault("config", {"displayModeBar": False})
    return _orig_plotly(fig, *a, **kw)


if not getattr(st.plotly_chart, "_nobar", False):
    _plotly_no_toolbar._nobar = True
    st.plotly_chart = _plotly_no_toolbar

LEVELS = ["High", "Medium", "Low"]
BRAND = "#4F46E5"
INK = "#1E1B4B"
GRID = "rgba(100,116,139,0.18)"
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color="#334155")


def dark() -> bool:
    return bool(st.session_state.get("dark_mode", False))


def pal() -> dict:
    """Colours that change with light / dark mode (charts and a few inline styles)."""
    if dark():
        return dict(ink="#F1F5F9", grid="rgba(165,180,252,0.20)", font="#CBD5E1", gap="#1B1946", axis="rgba(226,232,240,.6)")
    return dict(ink=INK, grid=GRID, font="#334155", gap="#ffffff", axis="rgba(30,27,75,.55)")


# --------------------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading model ...")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data(show_spinner="Preparing dashboard ...")
def get_dataset() -> dict:
    """Test-set students only (grouped split: none of these students were seen while learning)."""
    P = get_predictor()
    st_all = P.students
    S = st_all[st_all["split"] == "Test"].set_index("student_id").loc[P.test_feats.index].reset_index()
    S["id"] = S["student_id"].str.split("_").str[0]
    S["module"] = S["code_module"]
    S["intake"] = S["code_presentation"]
    S["group"] = S["code_module"] + " " + S["code_presentation"]
    agg = ex.aggregate_units(P.test_shap, P.columns)
    agg.index = P.test_feats.index
    labels = {k: P.units[k]["label"] for k in agg.columns}
    top = agg.idxmax(axis=1)
    S["main_risk_factor"] = [labels[k] if agg.iloc[i][k] > 0.003 else "-" for i, k in enumerate(top)]
    S["action"] = S["level"].map(ex.suggested_action)
    S["risk_pct"] = (S["risk"] * 100).clip(1, 99).round(0)
    return dict(inputs=P.test_inputs, feats=P.test_feats, students=S, shap_units=agg)


def pick_student(S: pd.DataFrame, ids: list, want, key: str, container=None):
    """Select a student by plain ID. If the same ID is enrolled in 2+ modules, a module box appears."""
    c = container or st
    sub = S[S["student_id"].isin(ids)]
    uniq = list(dict.fromkeys(sub["id"]))
    cur = S.loc[S["student_id"] == want, "id"].iloc[0] if (S["student_id"] == want).any() else uniq[0]
    pid = c.selectbox("Student ID", uniq, index=uniq.index(cur) if cur in uniq else 0, key=f"{key}_id")
    rows = sub[sub["id"] == pid]
    if len(rows) == 1:
        return rows["student_id"].iloc[0]
    lab = dict(zip(rows["student_id"], rows["module"] + " " + rows["intake"]))
    w = want if want in lab else rows["student_id"].iloc[0]
    return c.selectbox("Module (this student has more than one)", list(lab), index=list(lab).index(w),
                       format_func=lab.get, key=f"{key}_mod")


# --------------------------------------------------------------------------- styling
def _school_svg(fill: str, op: float = 1.0) -> str:
    """Original line-art school (building, flag, trees, books) as a URL-safe data URI."""
    f = fill.replace("#", "%23")
    body = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 420' fill='{f}' fill-opacity='{op}'>"
            "<rect x='250' y='170' width='400' height='190'/><polygon points='230,175 450,70 670,175'/>"
            "<rect x='405' y='20' width='6' height='60'/><polygon points='411,22 450,34 411,46'/>"
            "<rect x='400' y='260' width='100' height='100' fill='white'/>"
            "<circle cx='450' cy='140' r='26' fill='white'/><rect x='448' y='120' width='4' height='22' fill='%23888'/>"
            "<rect x='450' y='138' width='14' height='4' fill='%23888'/>"
            + "".join(f"<rect x='{x}' y='205' width='34' height='44' fill='white'/>" for x in (285, 340, 530, 585))
            + "".join(f"<rect x='{x}' y='285' width='34' height='44' fill='white'/>" for x in (285, 340, 530, 585))
            + "<rect x='215' y='360' width='470' height='14'/><rect x='190' y='374' width='520' height='14'/>"
            "<rect x='110' y='290' width='14' height='80'/><circle cx='117' cy='265' r='52'/>"
            "<rect x='776' y='290' width='14' height='80'/><circle cx='783' cy='265' r='52'/>"
            "<rect x='40' y='388' width='820' height='6'/></svg>")
    return "url(\"data:image/svg+xml;utf8," + body.replace("<", "%3C").replace(">", "%3E") + "\")"


# >>> SIDE MENU WIDTH (px): tukar nombor ni sahaja untuk besar / kecilkan menu <<<
SIDEBAR_W = 1340

CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 100% !important; padding-left: 3rem; padding-right: 3rem;}}
/* menu on the RIGHT-hand side */
[data-testid="stAppViewContainer"] {{flex-direction: row-reverse;}}
[data-testid="stSidebarResizeHandle"] {{display: none;}}
[data-testid="stSidebar"][aria-expanded="true"] {{border-left: 1px solid rgba(255,255,255,.12);}}
/* the open / close arrows point to the RIGHT-hand side too */
[data-testid="stSidebarCollapseButton"] button [data-testid="stIconMaterial"], [data-testid="stSidebarCollapseButton"] button svg {{transform: scaleX(-1);}}
[data-testid="stExpandSidebarButton"] {{position: fixed !important; left: auto !important; right: 3.6rem; top: .8rem; z-index: 999990;
  background: linear-gradient(120deg, #4338CA, #7C3AED) !important; border-radius: 10px; box-shadow: 0 6px 16px rgba(79,70,229,.35);}}
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] * {{color: #fff !important;}}
[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"], [data-testid="stExpandSidebarButton"] svg {{transform: scaleX(-1);}}
/* colourful page background with a faint school illustration */
[data-testid="stAppViewContainer"] {{background:
  {_school_svg("#4F46E5", 0.10)} no-repeat right bottom / min(62vw, 880px) auto fixed,
  radial-gradient(circle at 10% 6%, rgba(167,139,250,.38) 0, transparent 40%),
  radial-gradient(circle at 94% 10%, rgba(56,189,248,.32) 0, transparent 36%),
  linear-gradient(160deg, #E0E7FF 0%, #EDE9FE 50%, #DBEAFE 100%);}}
[data-testid="stHeader"] {{background: transparent;}}
/* sidebar */
[data-testid="stSidebar"] {{background:
  {_school_svg("#FFFFFF", 0.12)} no-repeat center bottom 10px / 94% auto,
  linear-gradient(185deg, #1E1B4B 0%, #312E81 55%, #5B21B6 100%);}}
[data-testid="stSidebar"] * {{color: #E0E7FF;}}
[data-testid="stSidebarHeader"] {{padding-top: 1.3rem; padding-bottom: .6rem;}}
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] {{height: 4.8rem !important; max-width: 100%;}}
/* wider menu */
[data-testid="stSidebar"] {{transition: width .3s ease, min-width .3s ease, opacity .25s ease !important; overflow: hidden;}}
[data-testid="stSidebar"][aria-expanded="true"] {{min-width: {SIDEBAR_W}px !important; width: {SIDEBAR_W}px !important;
  max-width: {SIDEBAR_W}px !important; opacity: 1;}}
[data-testid="stSidebar"][aria-expanded="true"] {{transform: none !important; visibility: visible;}}
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {{width: 100% !important;}}
/* closed menu: removed completely (no purple strip, no line, no text) */
[data-testid="stSidebar"][aria-expanded="false"], [data-testid="stSidebar"][aria-expanded="false"] * {{
  display: none !important; visibility: hidden !important; width: 0 !important; min-width: 0 !important; max-width: 0 !important;
  margin: 0 !important; padding: 0 !important; border: 0 !important; outline: 0 !important; box-shadow: none !important;
  background: none !important; transform: none !important; flex: 0 0 0 !important; opacity: 0 !important;}}
[data-testid="stSidebar"][aria-expanded="false"]::before, [data-testid="stSidebar"][aria-expanded="false"]::after {{display: none !important; content: none !important;}}
[data-testid="stSidebarResizeHandle"], [data-testid="stSidebar"] [role="separator"] {{display: none !important;}}
[data-testid="stApp"] {{overflow-x: hidden;}}
/* menu must stay fully inside the screen: the page area shrinks, the purple menu keeps its full width */
[data-testid="stAppViewContainer"] {{overflow: hidden !important; width: 100vw !important; max-width: 100vw !important;}}
[data-testid="stMain"] {{min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; max-width: 100% !important; overflow-x: hidden;}}
[data-testid="stSidebar"][aria-expanded="true"] {{flex: 0 0 min({SIDEBAR_W}px, 92vw) !important; margin-left: 0 !important; position: relative !important;
  min-width: min({SIDEBAR_W}px, 92vw) !important; width: min({SIDEBAR_W}px, 92vw) !important; max-width: min({SIDEBAR_W}px, 92vw) !important;}}
[data-testid="stSidebar"][aria-expanded="true"] > div, [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"] {{width: 100% !important; max-width: 100% !important;}}
[data-testid="stSidebarNav"] ul {{padding: 0 .5rem !important;}}
[data-testid="stSidebarNavLink"] {{font-size: .98rem; padding-right: .6rem !important;}}
[data-testid="stSidebarNavLink"] {{width: 100% !important; box-sizing: border-box !important; min-width: max-content;}}
[data-testid="stSidebarNavLink"] span {{white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important;}}
[data-testid="stSidebarNav"] ul {{padding: 0 .35rem;}}
[data-testid="stSidebarNavLink"] {{background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.16);
  border-radius: 12px; padding: .6rem .9rem; margin-bottom: 9px; font-weight: 600;
  transition: transform .15s, background .15s;}}
[data-testid="stSidebarNavLink"]:hover {{background: rgba(255,255,255,.20); transform: translateX(3px);}}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 65%, #06B6D4 140%);
  border-color: rgba(255,255,255,.40); box-shadow: 0 6px 16px rgba(0,0,0,.30);}}
[data-testid="stSidebarNavLink"][aria-current="page"] * {{color: #fff !important;}}
/* cards */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"],
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stPlotlyChart"]):not(:has(> [data-testid="stLayoutWrapper"])) {{
  background: linear-gradient(180deg, rgba(255,255,255,.95) 0%, rgba(238,242,255,.95) 100%) !important;
  border: 2px solid #818CF8 !important; border-radius: 16px;
  box-shadow: 0 8px 22px rgba(79,70,229,.15);}}
[data-testid="stExpander"] {{background: rgba(255,255,255,.9); border-radius: 14px; border: 2px solid #818CF8;}}
[data-testid="stMain"] h3 {{border-left: 5px solid #7C3AED; padding-left: .65rem; margin-top: .6rem;}}
.hero {{background: radial-gradient(circle at 85% 20%, rgba(255,255,255,.22) 0, transparent 45%),
        linear-gradient(120deg, #4338CA 0%, #7C3AED 55%, #0EA5E9 115%); color: #fff;
       border-radius: 18px; padding: 22px 30px; margin-bottom: 18px;
       box-shadow: 0 12px 28px rgba(79,70,229,.32);}}
.hero {{position: relative; overflow: hidden;}}
.hero::after {{content: ""; position: absolute; right: 18px; bottom: -6px; width: 300px; height: 135px;
  background: {_school_svg("#FFFFFF")} no-repeat right bottom / contain; opacity: .20; pointer-events: none;}}
.hero h1 {{color:#fff; font-size: 1.9rem; margin: 0; padding: 0; font-weight: 800; letter-spacing: -.01em;}}
.hero p {{color: rgba(255,255,255,.92); margin: 6px 0 0 0; font-size: .95rem; max-width: 78%;}}
.kpi {{background: color-mix(in srgb, var(--c, #4F46E5) 16%, #ffffff); border: 1px solid color-mix(in srgb, var(--c, #4F46E5) 35%, #ffffff);
      border-top: 6px solid var(--c, #4F46E5); border-radius: 14px;
      padding: 14px 18px; height: 100%; box-shadow: 0 8px 20px rgba(79,70,229,.14);}}
.kpi .l {{font-size: .76rem; color: #475569; text-transform: uppercase; letter-spacing: .06em; font-weight: 700;}}
.kpi .v {{font-size: 2rem; font-weight: 800; line-height: 1.2; color: {INK};}}
.kpi .s {{font-size: .82rem; color: #475569;}}
.badge {{display:inline-block; padding: 4px 14px; border-radius: 999px; color: #fff; font-weight: 700;
        font-size: .9rem; background: var(--c, #888);}}
.hint {{border-left: 4px solid {BRAND}; background: linear-gradient(90deg, #E0E7FF, #F5F3FF); color: #312E81;
       padding: 10px 14px; border-radius: 8px; font-size: .95rem;}}
.legend-up {{color: #E5484D; font-weight: 700;}} .legend-down {{color: #3E7BFA; font-weight: 700;}}
.tag {{display:inline-block; background:#EEF2FF; color:#3730A3; border-radius: 8px; padding: 3px 10px;
      margin: 2px 4px 2px 0; font-size: .85rem;}}
/* boxes made with st.container(key="navbox_...") : same clear border as the other cards */
[class*="st-key-navbox"] {{background: linear-gradient(180deg, rgba(255,255,255,.95) 0%, rgba(238,242,255,.95) 100%);
  border: 2px solid #818CF8; border-radius: 16px; padding: .8rem 1.1rem; box-shadow: 0 8px 22px rgba(79,70,229,.15);}}
/* number input + / - buttons: clear and bold */
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{color: #1E1B4B !important; font-weight: 800;}}
[data-testid="stNumberInputStepDown"] *, [data-testid="stNumberInputStepUp"] * {{color: #1E1B4B !important; fill: #1E1B4B !important; stroke: #1E1B4B !important; opacity: 1 !important;}}
/* HTML tables (ui.table): header and first column (attribute) differ from the values */
.ytable-wrap {{display: block; overflow-x: auto; overflow-y: hidden; border: 2px solid #818CF8; border-radius: 14px; background: #fff;
  box-shadow: 0 8px 22px rgba(79,70,229,.15); margin: 0 0 .6rem 0; padding: 0; line-height: 0; height: fit-content;
  scrollbar-width: thin;}}
.ytable-wrap p, .ytable-wrap style {{margin: 0 !important; padding: 0 !important; display: none;}}
[data-testid="stMarkdownContainer"]:has(.ytable-wrap) {{margin: 0 !important; padding: 0 !important;}}
.ytable {{border-collapse: collapse; width: 100%; margin: 0; font-size: .92rem; line-height: 1.35; display: table;}}
[data-testid="stMarkdownContainer"] .ytable-wrap table.ytable {{margin: 0 !important; margin-bottom: 0 !important;}}
.ytable tr:last-child td {{border-bottom: none !important;}}
.ytable th {{background: linear-gradient(120deg, #4338CA, #7C3AED); color: #fff; text-align: left; padding: 10px 14px;
  font-weight: 700; white-space: nowrap; border: none; position: sticky; top: 0;}}
.ytable td {{background: #fff; color: #1E293B; padding: 9px 14px; border-bottom: 1px solid #E0E7FF; border-left: 1px solid #E0E7FF;}}
.ytable td.col0 {{background: #E0E7FF; color: #1E1B4B; font-weight: 700; border: none !important; border-bottom: 1px solid #A5B4FC !important; border-right: 2px solid #818CF8 !important;}}
.ytable tr:hover td:not(.col0) {{background: #F5F3FF;}}
</style>
"""



DARK_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{background:
  {_school_svg("#A5B4FC", 0.10)} no-repeat right bottom / min(62vw, 880px) auto fixed,
  radial-gradient(circle at 10% 6%, rgba(124,58,237,.30) 0, transparent 42%),
  radial-gradient(circle at 94% 10%, rgba(14,165,233,.22) 0, transparent 38%),
  linear-gradient(160deg, #0E0D2B 0%, #181545 55%, #0B1A3B 100%) !important;}}
[data-testid="stMain"] :is(p, label, li, h1, h2, h3, h4, h5, h6, span, summary) {{color: #E5E7FF;}}
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stMain"] [data-testid="stCaptionContainer"] * {{color: #A5B4FC;}}
[data-testid="stToolbar"] *, [data-testid="stHeader"] * {{color: #E5E7FF;}}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stForm"],
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stPlotlyChart"]):not(:has(> [data-testid="stLayoutWrapper"])) {{
  background: linear-gradient(180deg, rgba(40,37,98,.96) 0%, rgba(27,25,70,.96) 100%) !important;
  border: 2px solid #C7D2FE !important;
  box-shadow: 0 0 0 1px rgba(199,210,254,.45), 0 0 14px rgba(165,180,252,.35), 0 8px 22px rgba(0,0,0,.40);}}
[data-testid="stExpander"] {{background: rgba(40,37,98,.95); border: 2px solid #C7D2FE;}}
[data-testid="stExpander"] details {{background: #1B1946 !important; border-color: #C7D2FE !important; border-radius: 12px;}}
[data-testid="stExpander"] summary {{background: #2E2A6E !important; border-radius: 10px;}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{color: #FFFFFF !important; font-weight: 700;}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{background: transparent !important;}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] :is(p, li, span, strong) {{color: #F1F5F9 !important;}}
[data-testid="stExpander"], [data-testid="stExpander"] :is(p, li, span, div, summary, label, strong, em, code, h1, h2, h3, h4, h5, h6) {{color: #E5E7FF !important;}}
[data-testid="stExpander"] [data-testid="stCaptionContainer"] * {{color: #C7D2FE !important;}}
[data-testid="stMain"] code {{background: rgba(129,140,248,.25) !important; color: #FDE68A !important;}}
.kpi {{background: color-mix(in srgb, var(--c, #4F46E5) 30%, #1B1946); border-color: color-mix(in srgb, var(--c, #4F46E5) 60%, #1B1946);
      box-shadow: 0 8px 20px rgba(0,0,0,.35);}}
.kpi .l, .kpi .s {{color: #C7D2FE;}} .kpi .v {{color: #FFFFFF;}}
.hint {{background: linear-gradient(90deg, #26235C, #2E2A6E); color: #E0E7FF; border-left-color: #818CF8;}}
.idcard {{background: linear-gradient(120deg, #26235C, #2E2A6E); border-color: #4F46E5; border-left-color: #A78BFA;}}
.idcard .t {{color: #C7D2FE;}} .idcard .r, .idcard .p {{color: #F1F5F9;}}
.tag {{background: #2E2A6E; color: #C7D2FE;}}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], textarea {{
  background-color: #25225A !important; color: #E5E7FF !important; border-color: #4F46E5 !important;}}
input {{color: #E5E7FF !important; -webkit-text-fill-color: #E5E7FF !important;}}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {{background: #25225A !important;}}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {{color: #E5E7FF !important;}}
[data-testid="stBaseButton-secondary"], [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{
  background: #2A2766 !important; border-color: #4F46E5 !important; color: #E5E7FF !important;}}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{color: #FFFFFF !important;}}
[data-testid="stNumberInputStepDown"] *, [data-testid="stNumberInputStepUp"] * {{color: #FFFFFF !important; fill: #FFFFFF !important; stroke: #FFFFFF !important; opacity: 1 !important;}}
[data-baseweb="tab"] {{color: #C7D2FE !important;}}
input:-webkit-autofill, input:-webkit-autofill:hover, input:-webkit-autofill:focus {{-webkit-box-shadow: 0 0 0 1000px #2A2766 inset !important; box-shadow: 0 0 0 1000px #2A2766 inset !important; -webkit-text-fill-color: #E5E7FF !important; caret-color: #E5E7FF;}}
input::placeholder, textarea::placeholder {{color: #A5B4FC !important; -webkit-text-fill-color: #A5B4FC !important; opacity: 1;}}
[data-baseweb="select"] > div, [data-baseweb="select"] > div > div {{background-color: #25225A !important;}}
[data-baseweb="select"] div, [data-baseweb="select"] svg {{color: #E5E7FF !important; fill: #E5E7FF;}}
[data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"], [data-testid="stTextAreaRootElement"] {{
  background: #25225A !important; border-color: #4F46E5 !important;}}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"] {{background: #25225A !important; border-color: #4F46E5 !important;}}
[data-testid="stSelectbox"] [role="group"] *, [data-testid="stMultiSelect"] [role="group"] button {{color: #E5E7FF !important;}}
[role="listbox"], [role="listbox"] > *, [data-testid="stSelectboxVirtualDropdown"] {{background: #25225A !important; color: #E5E7FF !important;}}
[role="option"] {{background: #25225A !important; color: #E5E7FF !important;}} [role="option"]:hover {{background: #3730A3 !important;}}
[data-baseweb="tag"] {{background-color: #4F46E5 !important;}} [data-baseweb="tag"] * {{color: #fff !important;}}
[data-testid="stDataFrame"] {{filter: invert(.93) hue-rotate(180deg);}}
/* message boxes (info / success / warning / error) readable on the dark background */
[data-testid="stAlert"] {{background: rgba(37,34,90,.95) !important; border: 1px solid #4F46E5; border-left: 6px solid #818CF8;}}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{border-left-color: #2FB380; background: rgba(21,66,58,.92) !important;}}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {{border-left-color: #F5A524; background: rgba(76,52,20,.92) !important;}}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{border-left-color: #E5484D; background: rgba(82,28,40,.92) !important;}}
[data-testid="stAlert"], [data-testid="stAlert"] * {{color: #F1F5F9 !important;}}
[data-testid="stAlert"] svg {{fill: #F1F5F9 !important;}}
[data-testid="stAlertContainer"], [data-testid="stAlert"] [data-testid^="stAlertContent"] {{background: transparent !important;}}
[data-testid="stBaseLinkButton-secondary"] {{background: #2A2766 !important; border-color: #4F46E5 !important;}}
[data-testid="stBaseLinkButton-secondary"], [data-testid="stBaseLinkButton-secondary"] * {{color: #E5E7FF !important;}}
[data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button {{background: #2A2766 !important; border-color: #4F46E5 !important; color: #E5E7FF !important;}}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"], [data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-pressed="true"], [data-testid="stButtonGroup"] button[kind$="Active"] {{background: #4F46E5 !important; color: #FFFFFF !important;}}
[data-testid="stButtonGroup"] button *, [data-testid="stSegmentedControl"] button * {{color: inherit !important;}}
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] li::marker {{color: #FFFFFF;}}
[class*="st-key-navbox"] {{background: linear-gradient(180deg, rgba(40,37,98,.96) 0%, rgba(27,25,70,.96) 100%);
  border: 2px solid #C7D2FE; box-shadow: 0 0 0 1px rgba(199,210,254,.45), 0 0 14px rgba(165,180,252,.35), 0 8px 22px rgba(0,0,0,.40);}}
.ytable-wrap {{background: #1B1946; border-color: #C7D2FE;
  box-shadow: 0 0 0 1px rgba(199,210,254,.45), 0 8px 22px rgba(0,0,0,.40);}}
.ytable th {{background: linear-gradient(120deg, #6366F1, #8B5CF6); color: #FFFFFF;}}
.ytable td {{background: #1B1946; color: #F1F5F9; border-color: #3B3796;}}
.ytable td.col0 {{background: #3730A3; color: #FFFFFF; border: none !important; border-bottom: 1px solid rgba(199,210,254,.55) !important; border-right: 2px solid #A5B4FC !important;}}
.ytable tr:hover td:not(.col0) {{background: #26235C;}}
</style>
"""


def inject_css():
    st.markdown(CSS + (DARK_CSS if dark() else ""), unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str | None = None):
    style = f' style="--c:{color}"' if color else ""
    st.markdown(f'<div class="kpi"{style}><div class="l">{label}</div><div class="v">{value}</div>'
                f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def badge(level: str) -> str:
    return (f'<span class="badge" style="--c:{ex.RISK_COLOR[level]};color:{ex.RISK_TEXT[level]}">'
            f'{level} risk</span>')


def hint(text: str):
    st.markdown(f'<div class="hint">{text}</div>', unsafe_allow_html=True)


def table(df: pd.DataFrame, fmt=None, highlight=None, height: int | None = None):
    """Static HTML table. Header row and first column (the attribute) have their own colour,
    so they stand out from the values. fmt: a format string for numeric columns or a {column: format} dict.
    highlight: function(row) -> list of CSS styles (same as Styler.apply, axis=1)."""
    df = df.reset_index(drop=True)
    sty = df.style.hide(axis="index").format_index(escape="html", axis=1)
    txt = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if txt:
        sty = sty.format(escape="html", subset=txt)
    if isinstance(fmt, str):
        fmt = {c: fmt for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    if fmt:
        sty = sty.format(fmt)
    if highlight is not None:
        sty = sty.apply(highlight, axis=1)
    html = "".join(line.strip() for line in sty.to_html(table_attributes='class="ytable"').splitlines())
    limit = f"max-height:{height}px;overflow-y:auto;" if height else ""
    st.markdown(f'<div class="ytable-wrap" style="{limit}">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- charts
def _layout(fig, h=360, **kw):
    title = kw.pop("title", None)
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), font={**FONT, "color": pal()["font"]}, legend_font_color=pal()["font"],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(text=title, font=dict(size=16, color=pal()["ink"]), x=0.01) if title else None, **kw)
    _t = dict(color=pal()["ink"], size=13, weight=700)
    fig.update_xaxes(tickfont=_t, title_font=_t)
    fig.update_yaxes(tickfont=_t, title_font=_t)
    fig.update_layout(legend_font=dict(color=pal()["ink"], size=13, weight=700))
    return fig


def donut(counts: dict) -> go.Figure:
    labs = [l for l in LEVELS if counts.get(l, 0) > 0]
    fig = go.Figure(go.Pie(labels=[f"{l} risk" for l in labs], values=[counts[l] for l in labs], hole=.62,
                           marker=dict(colors=[ex.RISK_COLOR[l] for l in labs], line=dict(width=3, color=pal()["gap"])),
                           sort=False, textinfo="percent", outsidetextfont=dict(color=pal()["ink"], size=14), insidetextorientation="horizontal", textfont=dict(size=14, color=[ex.RISK_TEXT[l] for l in labs]),
                           hovertemplate="%{label}: %{value} students (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{sum(counts.values()):,}</b><br>students", showarrow=False,
                       font=dict(size=17, color=pal()["ink"]))
    return _layout(fig, 420, title="Risk level distribution", legend=dict(orientation="h", y=-0.05, x=0.15))


def reason_bar(df: pd.DataFrame, top: int = 7):
    """Home chart: for EVERY student in the selection take the one factor that raises that student's risk the most
    (largest positive SHAP value) and count students per factor. One colour, all risk levels together."""
    d = df[df["main_risk_factor"] != "-"]
    if d.empty:
        return None
    t = d["main_risk_factor"].value_counts().head(top).iloc[::-1]
    n_all = len(df)
    names = ["<br>".join(textwrap.wrap(str(i).split(" (")[0], 20)) for i in t.index]
    fig = go.Figure(go.Bar(
        y=names, x=[int(v) for v in t.values], orientation="h",
        marker=dict(color="#6366F1", cornerradius=5),
        text=[f"<b>{int(v):,}</b> ({v / n_all:.0%})" for v in t.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:,} students<extra></extra>"))
    fig.update_xaxes(title="Number of students", gridcolor=pal()["grid"], range=[0, float(t.max()) * 1.28])
    fig.update_yaxes(automargin=True)
    return _layout(fig, 420, title="What pushes students towards dropping out most?"
                   "<br><span style='font-size:12px;font-weight:400'>Top factor raising each student's risk (% of all students, every risk level)</span>"
                   ).update_layout(margin=dict(l=10, r=45, t=70, b=10), showlegend=False)


def group_bars(shap_units: pd.DataFrame, levels: pd.Series, units: dict, level: str, top: int = 8) -> go.Figure:
    """Global SHAP for ONE risk group: average effect of each factor on the students in that group."""
    rows = shap_units[levels.reindex(shap_units.index).values == level]
    m = (rows.mean() * 100)
    m = m.reindex(m.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=m.values, y=[units[k]["label"].split(" (")[0] for k in m.index], orientation="h",
        marker=dict(color=[ex.UP_COLOR if v > 0 else ex.DOWN_COLOR for v in m.values], cornerradius=5),
        text=[f"{v:+.1f}" for v in m.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>average effect: %{x:+.1f} points<extra></extra>"))
    lim = max(float(m.abs().max()), 1) * 1.3
    fig.add_vline(x=0, line=dict(color=pal()["axis"], width=1.5))
    fig.update_xaxes(range=[-lim, lim], title="Average effect on risk (percentage points)", gridcolor=pal()["grid"], zeroline=False)
    fig.update_yaxes(automargin=True)
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="right",
                       text="<b>◀ lowers risk</b>  ", font=dict(color=ex.DOWN_COLOR, size=13))
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="left",
                       text="  <b>raises risk ▶</b>", font=dict(color=ex.UP_COLOR, size=13))
    return _layout(fig, 110 + 36 * len(m), title="Average effect of each factor on this group").update_layout(
        margin=dict(l=10, r=45, t=75, b=10))


def group_dots(shap_units: pd.DataFrame, feats: pd.DataFrame, levels: pd.Series, units: dict, level: str,
               top: int = 8, n: int = 400, seed: int = 0) -> go.Figure:
    """Dot plot for ONE risk group. One dot = one student. x = how much the factor moved that student's risk,
    colour = the student's own value of the factor (blue = low, red = high)."""
    ids = shap_units.index[levels.reindex(shap_units.index).values == level]
    order = shap_units.loc[ids].abs().mean().sort_values(ascending=False).index
    keys = [k for k in order if units[k]["kind"] != "category"][:top][::-1]
    rng = np.random.default_rng(seed)
    sample = ids if len(ids) <= n else ids[rng.choice(len(ids), n, replace=False)]
    fig = go.Figure()
    for j, k in enumerate(keys):
        v = pd.to_numeric(feats.loc[sample, k]).astype(float).values
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        if hi <= lo:
            lo, hi = float(v.min()), float(v.max())
        c = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), .5)
        fig.add_trace(go.Scatter(
            x=shap_units.loc[sample, k].values * 100, y=j + rng.uniform(-.3, .3, len(sample)), mode="markers",
            marker=dict(size=7, opacity=.72, color=c, cmin=0, cmax=1,
                        colorscale=[[0, "#3E7BFA"], [.5, "#CBD5E1"], [1, "#E5484D"]], showscale=(j == len(keys) - 1),
                        colorbar=dict(title=dict(text="Student's value", side="top"), tickvals=[0, 1],
                                      ticktext=["Low", "High"], len=.55, thickness=12, x=1.02, tickfont=dict(color=pal()["font"]),
                                      title_font=dict(color=pal()["font"]))),
            customdata=np.c_[v], hovertemplate=units[k]["label"] + "<br>this student's value: %{customdata[0]:.2f}"
                                               "<br>effect on risk: %{x:+.1f} points<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line=dict(color=pal()["axis"], width=1.5))
    fig.update_yaxes(tickvals=list(range(len(keys))),
                     ticktext=["<br>".join(textwrap.wrap(units[k]["label"].split(" (")[0], 24)) for k in keys],
                     automargin=True, gridcolor=pal()["grid"])
    fig.update_xaxes(title="Effect on this student's risk (percentage points)", gridcolor=pal()["grid"], zeroline=False)
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="right",
                       text="<b>◀ lowers risk</b>  ", font=dict(color=ex.DOWN_COLOR, size=13))
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="left",
                       text="  <b>raises risk ▶</b>", font=dict(color=ex.UP_COLOR, size=13))
    note = f" (sample of {len(sample):,} of {len(ids):,})" if len(ids) > n else f" ({len(ids):,} students)"
    return _layout(fig, 120 + 52 * len(keys), title="Every student, factor by factor" + note).update_layout(
        margin=dict(l=10, r=80, t=80, b=10))


def course_stack(df: pd.DataFrame) -> go.Figure:
    t = df.groupby(["group", "level"]).size().unstack(fill_value=0).reindex(columns=LEVELS, fill_value=0)
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for l in LEVELS:
        fig.add_bar(y=t.index, x=t[l], name=f"{l} risk", orientation="h",
                    marker=dict(color=ex.RISK_COLOR[l], line=dict(width=1.5, color=pal()["gap"])),
                    hovertemplate="%{y}<br>" + l + ": %{x} students<extra></extra>")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
    fig.update_xaxes(title="Students", gridcolor=pal()["grid"])
    return _layout(fig, 130 + 30 * len(t), title="Risk level by module and intake").update_layout(
        margin=dict(l=10, r=10, t=100, b=10))


def risk_meter(prob: float) -> go.Figure:
    fig = go.Figure()
    for lo, hi, l in ((0, cfg.LOW_MAX, "Low"), (cfg.LOW_MAX, cfg.HIGH_MIN, "Medium"), (cfg.HIGH_MIN, 1, "High")):
        fig.add_bar(x=[(hi - lo) * 100], y=[""], base=[lo * 100], orientation="h", marker_color=ex.RISK_COLOR[l],
                    showlegend=False, hoverinfo="skip", text=l, textposition="inside",
                    insidetextfont=dict(color=ex.RISK_TEXT[l], size=13))
    x = min(max(prob * 100, 0.8), 99.2)
    fig.add_vline(x=x, line=dict(width=4, color=pal()["ink"]))
    fig.add_annotation(x=min(max(prob * 100, 4), 96), y=1.0, yref="paper", text=f"▼ <b>{ex.pct(prob)}</b>",
                       showarrow=False, yanchor="bottom", font=dict(size=15, color=pal()["ink"]))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", gridcolor=pal()["grid"])
    fig.update_yaxes(visible=False)
    fig.update_layout(barmode="stack")
    return _layout(fig, 130, bargap=0.1).update_layout(margin=dict(l=10, r=10, t=45, b=10), title=None)


def local_bar(tbl: pd.DataFrame, base: float, prob: float, top: int = 10) -> go.Figure:
    """Local SHAP: right (red) = raises risk, left (blue) = lowers risk."""
    t = tbl.head(top).copy()
    rest = tbl.iloc[top:]
    names = [f"{r.feature}  =  {r.value}" for r in t.itertuples()]
    pts = list(t["points"])
    if len(rest):
        names.append(f"All other factors ({len(rest)})")
        pts.append(float(rest["points"].sum()))
    names, pts = names[::-1], pts[::-1]
    fig = go.Figure(go.Bar(
        x=pts, y=names, orientation="h",
        marker=dict(color=[ex.UP_COLOR if v > 0 else ex.DOWN_COLOR for v in pts], cornerradius=5),
        text=[f"{v:+.1f}" for v in pts], textposition="outside", cliponaxis=False, textfont=dict(color=pal()["ink"]),
        hovertemplate="%{y}<br>effect on risk: %{x:+.1f} percentage points<extra></extra>"))
    lim = max(abs(v) for v in pts) * 1.3 + 1
    fig.add_vline(x=0, line=dict(color=pal()["axis"], width=1.5))
    fig.update_xaxes(range=[-lim, lim], title="Effect on this student's dropout risk (percentage points)",
                     gridcolor=pal()["grid"], zeroline=False)
    fig.update_yaxes(automargin=True)
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="right",
                       text="<b>◀ lowers risk</b>  ", font=dict(color=ex.DOWN_COLOR, size=14))
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", yanchor="bottom", showarrow=False, xanchor="left",
                       text="  <b>raises risk ▶</b>", font=dict(color=ex.UP_COLOR, size=14))
    return _layout(fig, 90 + 38 * len(names), title=None).update_layout(margin=dict(l=10, r=40, t=40, b=10))


def importance_table(shap_units: pd.DataFrame, units: dict) -> pd.DataFrame:
    imp = (shap_units.abs().mean() * 100).sort_values(ascending=False)
    return pd.DataFrame({"key": imp.index, "feature": [units[k]["label"] for k in imp.index],
                         "importance": imp.values})


def global_bar(imp: pd.DataFrame, top: int = 15) -> go.Figure:
    t = imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                           marker=dict(color=BRAND, cornerradius=5),
                           text=[f"{v:.1f}" for v in t["importance"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>average effect: %{x:.2f} points<extra></extra>"))
    fig.update_xaxes(title="Average effect on risk (percentage points, ignoring direction)", gridcolor=pal()["grid"])
    fig.update_yaxes(automargin=True)
    return _layout(fig, 90 + 32 * len(t), title="Which factors matter most overall").update_layout(
        margin=dict(l=10, r=40, t=50, b=10))


def beeswarm(shap_units: pd.DataFrame, feats: pd.DataFrame, units: dict, order: list[str], top=10,
             n=700, seed=0) -> go.Figure:
    """Each dot = one student. x = effect on risk, colour = the student's value (blue low .. red high)."""
    rng = np.random.default_rng(seed)
    feats = feats.loc[shap_units.index]
    idx = rng.choice(len(shap_units), min(n, len(shap_units)), replace=False)
    fig = go.Figure()
    keys = [k for k in order if units[k]["kind"] != "category"][:top][::-1]
    for j, k in enumerate(keys):
        v = pd.to_numeric(feats[k]).iloc[idx].astype(float).values
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        c = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), .5)
        fig.add_trace(go.Scatter(
            x=shap_units[k].iloc[idx].values * 100, y=j + rng.uniform(-.28, .28, len(idx)), mode="markers",
            marker=dict(size=6, opacity=.75, color=c, cmin=0, cmax=1,
                        colorscale=[[0, "#3E7BFA"], [.5, "#CBD5E1"], [1, "#E5484D"]], showscale=(j == len(keys) - 1),
                        colorbar=dict(title="Value", tickvals=[0, 1], ticktext=["Low", "High"], len=.5, thickness=12)),
            customdata=np.c_[v], hovertemplate=units[k]["label"] + "<br>value: %{customdata[0]:.2f}"
                                               "<br>effect: %{x:+.1f} points<extra></extra>", showlegend=False))
    fig.add_vline(x=0, line=dict(color=pal()["axis"]))
    fig.update_yaxes(tickvals=list(range(len(keys))), ticktext=[units[k]["label"] for k in keys], automargin=True)
    fig.update_xaxes(title="Effect on dropout risk (percentage points):  ◀ lowers   |   raises ▶", gridcolor=pal()["grid"])
    return _layout(fig, 80 + 42 * len(keys), title="Direction of each factor (one dot = one student)").update_layout(
        margin=dict(l=10, r=10, t=50, b=10))


def direction_table(shap_units: pd.DataFrame, feats: pd.DataFrame, units: dict, order: list[str]) -> pd.DataFrame:
    """Plain-language direction: does a HIGHER value raise or lower the risk?"""
    feats = feats.loc[shap_units.index]
    rows = []
    for k in order:
        u = units[k]
        if u["kind"] == "category":
            rows.append((u["label"], "Depends on the category (see individual explanations)"))
            continue
        v = pd.to_numeric(feats[k]).astype(float)
        if v.std() == 0 or shap_units[k].std() == 0:
            continue
        r = float(np.corrcoef(v, shap_units[k])[0, 1])
        if abs(r) < .1:
            txt = "Mixed / weak direction"
        elif u["kind"] == "binary":
            txt = ("'Yes' raises risk" if r > 0 else "'Yes' lowers risk")
        else:
            txt = ("Higher value raises risk" if r > 0 else "Higher value lowers risk")
        rows.append((u["label"], txt))
    return pd.DataFrame(rows, columns=["Factor", "How it works"])


def confusion_chart(tn: int, fp: int, fn: int, tp: int) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    txt = [[f"<b>{tn}</b><br>cleared", f"<b>{fp}</b><br>false alarm"],
           [f"<b>{fn}</b><br>missed at-risk", f"<b>{tp}</b><br>at-risk caught"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted: Not at-risk", "Predicted: At-risk"],
                               y=["Actual: Not at-risk", "Actual: At-risk"], text=txt, texttemplate="%{text}",
                               colorscale=[[0, "#EEF2FF"], [1, BRAND]], showscale=False, xgap=4, ygap=4,
                               hoverinfo="skip", textfont=dict(size=14)))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 330, title="Confusion matrix (test set)")


# --------------------------------------------------------------------------- explanation + follow-up blocks
def explanation_block(res: Result, key: str, top: int = 10, summary: bool = True, actions: bool = True):
    """Risk meter + local SHAP chart + words + recommended actions (Dashboard and Live pages)."""
    if summary:
        c1, c2 = st.columns([1, 2])
        with c1:
            kpi("Predicted dropout risk", ex.pct(res.prob), f"Average student: {ex.pct(res.base)}",
                ex.RISK_COLOR[res.level])
            st.markdown(f"<div style='margin-top:10px'>{badge(res.level)}</div>", unsafe_allow_html=True)
        with c2:
            with st.container(border=True):
                st.plotly_chart(risk_meter(res.prob), width="stretch", key=f"{key}_meter")
                st.caption(f"Low < {cfg.LOW_MAX:.0%}  |  Medium {cfg.LOW_MAX:.0%}-{cfg.HIGH_MIN:.0%}  |  "
                           f"High >= {cfg.HIGH_MIN:.0%}")
    st.markdown("#### Why this risk? (local SHAP explanation)")
    r_, b_ = round(min(max(res.prob * 100, 1), 99)), round(res.base * 100)
    d_ = r_ - b_
    hint(f"An average student has <b>{b_}%</b> risk. Each bar is one fact about this student: "
         "<span class='legend-up'>red adds</span> risk points, <span class='legend-down'>blue removes</span> them. "
         f"So: <b>{b_}% {'+' if d_ >= 0 else '-'} {abs(d_)} points = {r_}%</b>.")
    with st.container(border=True):
        st.plotly_chart(local_bar(res.table, res.base, res.prob, top), width="stretch", key=f"{key}_local")
    st.markdown(ex.narrative(res.prob, res.base, res.table))
    if actions:
        st.markdown("#### Recommended support")
        for a in ex.recommend(res.table, res.level):
            st.markdown(f"- {a}")


def result_card(res: Result, profile: dict, key: str):
    """'Predicted dropout risk for student ...' header used on the Live Prediction page."""
    c1, c2 = st.columns([1.25, 1.6])
    color = ex.RISK_COLOR[res.level]
    shown = lambda v: v if str(v).strip() else "-"
    with c1:
        st.markdown(
            f'<div class="kpi" style="--c:{color}"><div class="l">Predicted dropout risk for student</div>'
            f'<div style="font-size:1.05rem;line-height:1.75;margin:6px 0 8px 0;color:{pal()["ink"]}">'
            f'<b>Name:</b> {shown(profile.get("name", ""))}<br><b>Student ID:</b> {shown(profile.get("sid", ""))}<br>'
            f'<b>Age:</b> {shown(profile.get("age", ""))}</div>'
            f'{badge(res.level)} &nbsp; <span style="font-size:2rem;font-weight:800;color:{pal()["ink"]}">{ex.pct(res.prob)}</span>'
            f'<div class="s">Average student: {ex.pct(res.base)}</div></div>', unsafe_allow_html=True)
    with c2, st.container(border=True):
        st.plotly_chart(risk_meter(res.prob), width="stretch", key=f"{key}_meter")
        st.caption(f"Low < {cfg.LOW_MAX:.0%}  |  Medium {cfg.LOW_MAX:.0%}-{cfg.HIGH_MIN:.0%}  |  High >= {cfg.HIGH_MIN:.0%}")


def approval_follow_up(res: Result, key: str, name: str = "", phone: str = "", email: str = ""):
    """Live prediction follow-up (contacts come from the form).
    High   -> admin approves the message -> WhatsApp link
    Medium -> admin approves the message -> e-mail link
    Low    -> monitoring list only
    """
    level, prob = res.level, res.prob
    st.subheader("Recommended action")
    if level == "Low":
        st.success("Low risk: no message needed. Keep monitoring:")
        for a in ex.monitoring_plan(res.table):
            st.markdown(f"- {a}")
        return
    wa = level == "High"
    contact = phone if wa else email
    reasons = [f"{r.feature} = {r.value}" for r in res.table[res.table.points > 0.3].head(3).itertuples()]
    actions = [a.replace("**", "") for a in ex.recommend(res.table, level)]
    subject, body = notify.default_message(level, prob, reasons, actions, name)
    box = st.error if wa else st.warning          # High = red box, Medium = amber box (Low = green box above)
    box(f"{level} risk: follow-up by **{'WhatsApp' if wa else 'e-mail'}** "
        f"({contact.strip() or 'no ' + ('phone number' if wa else 'e-mail') + ' entered'}). Review the message, "
        "press **Approve**, then open the link. Nothing is sent automatically and nothing is stored.")
    msg = st.text_area("Message", value=body, height=260, key=f"{key}_msg_{level}_{int(prob * 1000)}_{name}")
    ok_key = f"{key}_approved"
    if st.button("Approve message", type="primary", icon=":material/verified:", key=f"{key}_approve"):
        st.session_state[ok_key] = (msg, contact)
    approved = st.session_state.get(ok_key) == (msg, contact)
    if not approved:
        st.info("The link appears after the message is approved. If you edit the message, approve it again.")
        return
    if not contact.strip():
        st.warning(f"Enter the {'WhatsApp number' if wa else 'e-mail address'} in the form above and predict again.")
        return
    st.success("Message approved.")
    if wa:
        link = notify.whatsapp_link(contact, msg)
        if link:
            st.link_button("Open WhatsApp with this message", link, type="primary", icon=":material/chat:")
        else:
            st.error("The phone number does not look valid.")
    else:
        g, m = notify.gmail_link(contact, subject, msg), notify.mailto_link(contact, subject, msg)
        if g:
            b1, b2, _ = st.columns([1, 1, 2])
            b1.link_button("Compose in Gmail", g, type="primary", icon=":material/mail:")
            b2.link_button("Open mail app", m, icon=":material/outgoing_mail:")
        else:
            st.error("The e-mail address does not look valid.")
