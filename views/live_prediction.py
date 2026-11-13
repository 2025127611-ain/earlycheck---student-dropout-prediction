import pandas as pd
import streamlit as st

from core import data_prep as dp
from core import ui

P = ui.get_predictor()
F = dp.INPUT_FIELDS

ui.header("Live Prediction", "Enter a student's details for an instant risk prediction. "
          "Nothing you type is saved or added to the dataset.")

# only the fields the RFE-selected features need (e.g. units with missed assessments)
need = set(P.required)


# ------------------------------------------------------------------ form
def widget(k: str):
    f, key = F[k], f"in_{k}"
    if key not in st.session_state:                      # defaults are set via session state only
        st.session_state[key] = (bool(f.get("default", False)) if f["type"] == "binary" else f["default"]) \
            if f["type"] != "select" else f.get("default", f["options"][0])
    if f["type"] == "binary":
        return int(st.checkbox(f["label"], key=key, help=f.get("help")))
    if f["type"] == "select":
        return st.selectbox(f["label"], f["options"], key=key, help=f.get("help"))
    if f["type"] == "int":
        return int(st.number_input(f["label"], f["min"], f["max"], step=1, key=key, help=f.get("help")))
    return float(st.number_input(f["label"], f["min"], f["max"], step=0.5, key=key, help=f.get("help")))


def reset_form():
    for k in [k for k in st.session_state if str(k).startswith(("in_", "live_fu"))] + ["live_result"]:
        st.session_state.pop(k, None)
    for k in ("in_sid", "in_name", "in_phone", "in_email"):      # text boxes must be emptied explicitly
        st.session_state[k] = ""


_, rb = st.columns([5, 1])
rb.button("Reset form", icon=":material/restart_alt:", on_click=reset_form, width="stretch")

AGE_BAND = lambda a: "0-35" if a < 35 else ("35-55" if a <= 55 else "55<=")

with st.form("live_form"):
    st.markdown("##### Student details")
    c1, c2, c3 = st.columns(3)
    st.session_state.setdefault("in_age", 25)
    sid_in = c1.text_input("Student ID", key="in_sid", placeholder="e.g. 570895")
    name_in = c2.text_input("Student name", key="in_name")
    age_in = int(c3.number_input("Age (years)", 15, 90, step=1, key="in_age"))
    c4, c5, _ = st.columns(3)
    phone_in = c4.text_input("WhatsApp number", key="in_phone", placeholder="012-3456789")
    email_in = c5.text_input("E-mail", key="in_email", placeholder="name@example.com")
    vals: dict = {}
    for section, keys in dp.INPUT_SECTIONS.items():
        keys = [k for k in keys if k in need and k != "age_band"]      # age band is worked out from the age above
        if not keys:
            continue
        st.markdown(f"##### {section}")
        cols = st.columns(3)
        for i, k in enumerate(keys):
            with cols[i % 3]:
                vals[k] = widget(k)
    go = st.form_submit_button("Predict risk", type="primary", icon=":material/bolt:")


def validate(v: dict) -> list[str]:
    err = []
    if {"assess_submitted", "assess_due"} <= v.keys() and v["assess_submitted"] > v["assess_due"]:
        err.append("Assessments submitted cannot be more than assessments due.")
    if {"n_late", "assess_submitted"} <= v.keys() and v["n_late"] > v["assess_submitted"]:
        err.append("Late submissions cannot be more than assessments submitted.")
    if {"clicks_recent", "total_clicks"} <= v.keys() and v["clicks_recent"] > v["total_clicks"]:
        err.append("Clicks in the last 4 weeks cannot be more than total clicks.")
    if {"active_days", "total_clicks"} <= v.keys() and v["active_days"] > 0 and v["total_clicks"] < v["active_days"]:
        err.append("Total clicks must be at least the number of active days.")
    return err


if go:
    errs = validate(vals)
    if errs:
        for e in errs:
            st.error(e)
        st.session_state.pop("live_result", None)
    else:
        if "age_band" in need:
            vals["age_band"] = AGE_BAND(age_in)
        res = P.explain_inputs(pd.DataFrame([vals]))[0]
        st.session_state["live_result"] = dict(
            res=res, profile=dict(sid=sid_in.strip(), name=name_in.strip(), age=age_in),
            phone=phone_in.strip(), email=email_in.strip())

# ------------------------------------------------------------------ result
out = st.session_state.get("live_result")
if out:
    res, prof = out["res"], out["profile"]
    st.divider()
    ui.result_card(res, prof, "live")
    ui.explanation_block(res, "live", summary=False, actions=False)
    st.divider()
    ui.approval_follow_up(res, key="live_fu", name=prof["name"], phone=out["phone"], email=out["email"])
    st.divider()
    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
        c1.markdown("##### Compare with existing students?\nSee how the model scored students in the test set.")
        if c2.button("Student list", icon=":material/dashboard:", width="stretch"):
            st.switch_page("views/home.py")
        if c3.button("SHAP explanation", icon=":material/insights:", width="stretch"):
            st.switch_page("views/shap_dashboard.py")
