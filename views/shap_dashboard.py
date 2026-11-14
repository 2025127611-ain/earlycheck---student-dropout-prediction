import pandas as pd
import streamlit as st

from core import data_prep as dp
from core import ui

P = ui.get_predictor()
D = ui.get_dataset()
S: pd.DataFrame = D["students"]            # test-set students only

ui.header("SHAP Explanation", f"Why did the model give each student this risk? Model: {P.meta['model_name']}.")
lv = st.multiselect("Risk level", ui.LEVELS, default=ui.LEVELS, key="shap_lv")      # one filter for both tabs
if not lv:
    st.warning("Select at least one risk level.")
    st.stop()
tab_local, tab_global = st.tabs(["Individual student", "By risk group"])

# ================================================================== LOCAL
with tab_local:
    pool = S[S["level"].isin(lv)].sort_values("risk", ascending=False)
    if pool.empty:
        st.warning("No students at this level.")
        st.stop()
    ids = pool["student_id"].tolist()
    pick_col, _ = st.columns([1, 1])
    sid = ui.pick_student(S, ids, st.session_state.get("selected_id"), "loc", pick_col)
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
        ui.table(show)

# ================================================================== GLOBAL (one section per selected risk group)
with tab_global:
    ui.hint("<b>Top graph (bars)</b>: the <i>average</i> effect of each factor on this group. Red bar = on average it "
            "<span class='legend-up'>raises</span> risk, blue bar = <span class='legend-down'>lowers</span> it.<br>"
            "<b>Bottom graph (dots)</b>: the same factors, but <i>one dot = one student</i>. Read each row like this: "
            "<b>where the dot sits</b> (left / right of the line) = did this factor lower or raise that student's risk; "
            "<b>dot colour</b> = that student's own value for the factor (<span class='legend-up'>red = high value</span>, "
            "<span class='legend-down'>blue = low value</span>).<br>"
            "<i>Example:</i> on the \"Average assessment score\" row, red dots on the left and blue dots on the right "
            "means students with a high score have lower risk, and students with a low score have higher risk. "
            "A wide spread of dots = the factor affects students very differently.")
    lv_map = S.set_index("student_id")["level"]
    for lvl in ui.LEVELS:
        if lvl not in lv or not (lv_map == lvl).any():
            continue
        st.markdown(f"### {ui.badge(lvl)} &nbsp; {(lv_map == lvl).sum():,} students", unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(ui.group_bars(D["shap_units"], lv_map, P.units, lvl), width="stretch", key=f"grp_{lvl}")
        with st.container(border=True):
            st.plotly_chart(ui.group_dots(D["shap_units"], D["feats"], lv_map, P.units, lvl), width="stretch",
                            key=f"dots_{lvl}")

st.divider()
with st.container(key="navbox_shap"):
    c1, c2, c3 = st.columns([2.2, 1, 1], vertical_alignment="center")
    c1.markdown("##### Done with this student?\nGo back to the full student list, or test a new student.")
    if c2.button("Back to student list", icon=":material/dashboard:", width="stretch"):
        st.switch_page("views/home.py")
    if c3.button("Try Live Prediction", type="primary", icon=":material/bolt:", width="stretch"):
        st.switch_page("views/live_prediction.py")
