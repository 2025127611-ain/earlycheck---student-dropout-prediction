import pandas as pd
import streamlit as st

from core import explain as ex
from core import ui

D = ui.get_dataset()
S: pd.DataFrame = D["students"]            # test-set students only

ui.header("Student Dropout Risk Prediction Dashboard",
          "EarlyCheck: A Learning Analytics-Based Early Warning System to Detect At-Risk Students and "
          "Recommend Timely Intervention")

# ------------------------------------------------------------------ filters
f1, f2, f3 = st.columns([1.2, 1, 1])
lv = f1.multiselect("Risk level", ui.LEVELS, default=ui.LEVELS)
mods = f2.multiselect("Module", sorted(S["module"].unique()), placeholder="All modules")
pres = f3.multiselect("Intake", sorted(S["intake"].unique()), placeholder="All intakes")
V = S[S["level"].isin(lv)]
if mods:
    V = V[V["module"].isin(mods)]
if pres:
    V = V[V["intake"].isin(pres)]
if V.empty:
    st.warning("No students match these filters.")
    st.stop()

# ------------------------------------------------------------------ KPIs
cnt = V["level"].value_counts().to_dict()
n = len(V)
k = st.columns(4)
with k[0]:
    ui.kpi("Students", f"{n:,}", f"Average risk {V['risk'].mean():.0%}", ui.BRAND)
for col, l in zip(k[1:], ui.LEVELS):
    with col:
        ui.kpi(f"{l} risk", f"{cnt.get(l, 0):,}", f"{cnt.get(l, 0) / n:.0%} of students", ex.RISK_COLOR[l])
st.write("")

# ------------------------------------------------------------------ charts
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.plotly_chart(ui.donut(cnt), width="stretch")
with c2, st.container(border=True):
    st.plotly_chart(ui.reason_bar(V), width="stretch")
with st.container(border=True):
    st.plotly_chart(ui.course_stack(V), width="stretch")

# ------------------------------------------------------------------ table
st.subheader("Student list")
st.caption("Tick a row, then press the button to see why the model gave that risk.")
show = V.sort_values("risk", ascending=False).reset_index(drop=True)
tbl = show[["id", "module", "intake", "risk_pct", "level", "main_risk_factor", "action"]]
tbl.columns = ["Student ID", "Module", "Intake", "Risk %", "Level", "Main risk factor", "Suggested action"]
event = st.dataframe(
    tbl, width="stretch", height=380, hide_index=True, on_select="rerun", selection_mode="single-row",
    column_config={"Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.0f%%")})
sel = event.selection.rows
if sel:
    sid = show.loc[sel[0], "student_id"]
    if st.button(f"Explain student ID {show.loc[sel[0], 'id']}", type="primary", icon=":material/insights:"):
        st.session_state["selected_id"] = sid
        st.switch_page("views/shap_dashboard.py")
st.download_button("Download list (CSV)", tbl.to_csv(index=False).encode(), "student_risk_list.csv", "text/csv")

st.divider()
with st.container(border=True):
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown("##### Want to check a student who is not in this list?\n"
                "Enter their details and get an instant risk prediction with the reasons.")
    if c2.button("Try Live Prediction", type="primary", icon=":material/bolt:", width="stretch"):
        st.switch_page("views/live_prediction.py")
