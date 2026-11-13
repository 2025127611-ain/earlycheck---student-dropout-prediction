import pandas as pd
import streamlit as st

from core import data_prep as dp
from core import ui

P = ui.get_predictor()
D = ui.get_dataset()
S: pd.DataFrame = D["students"]            # test-set students only

ui.header("SHAP Explanation", f"Why did the model give each student this risk? Model: {P.meta['model_name']}.")
tab_local, tab_global = st.tabs(["Individual student", "By risk group"])

# ================================================================== LOCAL
with tab_local:
    f1, f2 = st.columns([1.4, 1])
    lv = f1.multiselect("Risk level", ui.LEVELS, default=ui.LEVELS, key="loc_lv")
    pool = S[S["level"].isin(lv)].sort_values("risk", ascending=False)
    if pool.empty:
        st.warning("No students at this level.")
        st.stop()
    ids = pool["student_id"].tolist()
    sid = ui.pick_student(S, ids, st.session_state.get("selected_id"), "loc", f2)
    st.session_state["selected_id"] = sid
    res = P.explain_known(sid)
    row = S.index[S["student_id"] == sid][0]

    st.markdown(f"### Student ID: {S.loc[row, 'id']} &nbsp; {ui.badge(res.level)}", unsafe_allow_html=True)
    st.caption(f"Module: {S.loc[row, 'module']} {S.loc[row, 'intake']}")
    ui.explanation_block(res, "dash")

    with st.expander("Student details used by the model"):
        inp = D["inputs"].loc[sid, P.required]
        show = pd.DataFrame({"Field": [dp.INPUT_FIELDS[k]["label"] for k in P.required],
                             "Value": [str(v) for v in inp.values]})
        st.dataframe(show, hide_index=True, width="stretch")

# ================================================================== GLOBAL (one chart per risk group)
with tab_global:
    lv_map = S.set_index("student_id")["level"]
    for lvl in ui.LEVELS:
        if (lv_map == lvl).any():
            with st.container(border=True):
                st.plotly_chart(ui.group_bars(D["shap_units"], lv_map, P.units, lvl), width="stretch",
                                key=f"grp_{lvl}")

st.divider()
with st.container(border=True):
    c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
    c1.markdown("##### Done with this student?\nGo back to the full student list, or test a new student.")
    if c2.button("Back to student list", icon=":material/dashboard:", width="stretch"):
        st.switch_page("views/home.py")
    if c3.button("Try Live Prediction", type="primary", icon=":material/bolt:", width="stretch"):
        st.switch_page("views/live_prediction.py")
