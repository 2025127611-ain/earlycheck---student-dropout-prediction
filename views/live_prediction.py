import pandas as pd
import streamlit as st

from core import data_prep as dp
from core import notify
from core import ui

P = ui.get_predictor()
F = dp.INPUT_FIELDS

ui.header("Live Prediction", "Enter a student's details for an instant risk prediction. "
          "Nothing you type is saved or added to the dataset.")

# keep what the user typed: widget values are dropped by Streamlit when the page is left, so they are
# copied to a store and put back at the start of every run. Only "Reset form" clears the store.
STORE = st.session_state.setdefault("_live_store", {})
for _k, _v in STORE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# only the fields the RFE-selected features need (e.g. units with missed assessments)
need = set(P.required)


# ------------------------------------------------------------------ form
# Nothing is pre-filled: the user has to enter every field before a prediction is made.
def widget(k: str):
    f, key = F[k], f"in_{k}"
    if f["type"] == "binary":
        return int(st.checkbox(f["label"], key=key, help=f.get("help")))
    if f["type"] == "select":
        return st.selectbox(f["label"], f["options"], index=None, placeholder="Select ...", key=key, help=f.get("help"))
    if f["type"] == "int":
        return st.number_input(f["label"], f["min"], f["max"], value=None, step=1, placeholder="Enter a number",
                               key=key, help=f.get("help"))
    return st.number_input(f["label"], f["min"], f["max"], value=None, step=0.5, placeholder="Enter a number",
                           key=key, help=f.get("help"))


TEXT_KEYS = ("in_sid", "in_name", "in_age", "in_phone", "in_email")


def reset_form():
    STORE.clear()
    for k in [k for k in st.session_state if str(k).startswith(("in_", "live_fu"))] + ["live_result"]:
        if k in TEXT_KEYS:
            st.session_state[k] = ""                       # text boxes must be emptied explicitly
        elif str(k).startswith("in_") and F.get(str(k)[3:], {}).get("type") == "binary":
            st.session_state[k] = False
        elif str(k).startswith("in_"):
            st.session_state[k] = None
        else:
            st.session_state.pop(k, None)


_, rb = st.columns([5, 1])
rb.button("Reset form", icon=":material/restart_alt:", on_click=reset_form, width="stretch")

AGE_BAND = lambda a: "0-35" if a < 35 else ("35-55" if a <= 55 else "55<=")

with st.form("live_form", border=True):
    st.markdown("##### Student details")
    c1, c2, c3 = st.columns(3)
    sid_in = c1.text_input("Student ID", key="in_sid", placeholder="e.g. 570895", autocomplete="on")
    name_in = c2.text_input("Student name", key="in_name", placeholder="Full name", autocomplete="name")
    age_txt = c3.text_input("Age (years)", key="in_age", placeholder="e.g. 25", autocomplete="on")
    age_in = int(age_txt.strip()) if (age_txt or "").strip().isdigit() else None
    c4, c5, _ = st.columns(3)
    phone_in = c4.text_input("WhatsApp number", key="in_phone", placeholder="012-3456789", autocomplete="tel")
    email_in = c5.text_input("E-mail", key="in_email", placeholder="name@example.com", autocomplete="email")
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


# save the current form values (survives page change / dark mode / unfinished form)
for _k in [k for k in st.session_state if str(k).startswith("in_")]:
    STORE[_k] = st.session_state[_k]


def missing_fields() -> list[str]:
    miss = []
    if not (sid_in or "").strip():
        miss.append("Student ID")
    if not (name_in or "").strip():
        miss.append("Student name")
    if not (age_txt or "").strip():
        miss.append("Age")
    if not (phone_in or "").strip():
        miss.append("WhatsApp number")
    if not (email_in or "").strip():
        miss.append("E-mail")
    miss += [F[k]["label"] for k, v in vals.items() if v is None]
    return miss


def validate(v: dict) -> list[str]:
    err = []
    if age_in is None or not (15 <= age_in <= 90):
        err.append("Age must be a whole number between 15 and 90.")
    if not notify.normalise_phone(phone_in):
        err.append("The WhatsApp number does not look valid (example: 012-3456789).")
    if not notify.valid_email(email_in):
        err.append("The e-mail address does not look valid (example: name@example.com).")
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
    miss = missing_fields()
    if miss:
        st.session_state.pop("live_result", None)
        st.error("Please fill in the whole form before predicting. Still empty: " + ", ".join(miss) + ".")
    else:
        errs = validate(vals)
        if errs:
            st.session_state.pop("live_result", None)
            for e in errs:
                st.error(e)
        else:
            vals = {k: (int(v) if F[k]["type"] == "int" else v) for k, v in vals.items()}
            if "age_band" in need:
                vals["age_band"] = AGE_BAND(age_in)
            res = P.explain_inputs(pd.DataFrame([vals]))[0]
            st.session_state["live_result"] = dict(
                res=res, profile=dict(sid=sid_in.strip(), name=name_in.strip(), age=int(age_in)),
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
    with st.container(key="navbox_live"):
        c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
        c1.markdown("##### Compare with existing students?\nSee how the model scored students in the test set.")
        if c2.button("Student list", icon=":material/dashboard:", width="stretch"):
            st.switch_page("views/home.py")
        if c3.button("SHAP explanation", icon=":material/insights:", width="stretch"):
            st.switch_page("views/shap_dashboard.py")
