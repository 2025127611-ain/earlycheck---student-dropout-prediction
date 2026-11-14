import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import config as cfg
from core import ui

P = ui.get_predictor()
M = P.meta
R = M["results"]
best = R[M["model_name"]]
D = M["data_info"]
MAIN = M.get("main_split", "80:20")
IMB = {"smotenc": "SMOTENC on training folds only", "class_weight": "Class weights", "none": "None"}

ui.header("Model Performance", "Risk is a prediction to support early help, not a final judgement about any student.")

# ------------------------------------------------------------------ headline numbers
k = st.columns(4)
with k[0]:
    ui.kpi("Selected model", M["model_name"], "Best cross-validated ROC-AUC", ui.BRAND)
with k[1]:
    ui.kpi("Accuracy", f"{best['accuracy']:.1%}", "Unseen test set", "#0EA5A4")
with k[2]:
    ui.kpi("Recall (dropout)", f"{best['recall']:.1%}", f"{best['tp']} of {best['tp'] + best['fn']} dropouts found",
           "#F5A524")
with k[3]:
    ui.kpi("ROC-AUC", f"{best['roc_auc']:.3f}", "Unseen test set", "#7C3AED")
st.write("")

# ------------------------------------------------------------------ comparison
st.subheader(f"Model comparison ({MAIN} test set)")
BASE = "Decision Tree"
label = lambda n: n + " (baseline)" if n == BASE else n
rows = []
for name, m in R.items():
    rows.append({"Model": label(name),
                 "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
                 "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], f"Recall @ risk>={cfg.LOW_MAX:.0%}": m["recall_at_alert"],
                 "CV ROC-AUC (train)": m["cv_auc"]})
def _hl(color):
    chosen = label(M["model_name"])          # the row of the selected model is highlighted (values only)
    sty = f"background-color: {color}; color: #1E1B4B; font-weight: 700"
    return lambda r: [""] + [sty] * (len(r) - 1) if r.iloc[0] == chosen else [""] * len(r)


ui.table(pd.DataFrame(rows), fmt="{:.3f}", highlight=_hl("#C7D2FE"))
if BASE in R and M["model_name"] != BASE:
    st.caption(f"Baseline = Decision Tree (simplest model). All three models are tuned the same way (GridSearchCV). "
               f"{M['model_name']} improves ROC-AUC by {R[M['model_name']]['roc_auc'] - R[BASE]['roc_auc']:+.3f} "
               f"and accuracy by {R[M['model_name']]['accuracy'] - R[BASE]['accuracy']:+.3f} over the baseline.")

c1, c2 = st.columns([1.3, 1])
with c1, st.container(border=True):
    fig = go.Figure()
    colors = {"Decision Tree": "#F5A524", "Random Forest": "#0EA5A4", "XGBoost": ui.BRAND}
    for name, m in R.items():
        fig.add_bar(name=label(name), x=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                    y=[m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]],
                    marker=dict(color=colors.get(name, "#888"), cornerradius=4))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.0, x=0.0, yanchor="bottom"), margin=dict(t=80))
    fig.update_yaxes(range=[0.7, 1.0], gridcolor=ui.pal()["grid"])
    st.plotly_chart(ui._layout(fig, 340, title="Scores by model").update_layout(margin=dict(l=10, r=10, t=85, b=10)), width="stretch")
with c2, st.container(border=True):
    st.plotly_chart(ui.confusion_chart(best["tn"], best["fp"], best["fn"], best["tp"]), width="stretch")
real = best["tp"] + best["fn"]
st.caption(f"Example: {real:,} students in the test set really dropped out. The model flagged {best['tp']:,} of them "
           f"as High risk ({best['recall']:.1%}). If Medium risk students are also followed up, "
           f"{round(best['recall_at_alert'] * real):,} are caught ({best['recall_at_alert']:.1%}).")


# ------------------------------------------------------------------ split ratio comparison
SC = M.get("split_comparison")
if SC:
    st.subheader("Effect of the train / test ratio")
    st.caption(f"Same pipeline each time: student-level GroupShuffleSplit, encoder and RFE re-fitted on that split's training "
               f"students, SMOTENC on training data only, tuned settings reused. The dashboard uses {MAIN}.")
    rows = [{"Split (train:test)": lab, "Model": name, "Train rows": v["n_train"], "Test rows": v["n_test"],
             "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"],
             "ROC-AUC": m["roc_auc"]}
            for lab, v in SC.items() for name, m in v["models"].items()]
    sdf = pd.DataFrame(rows)
    pick = st.segmented_control("Metric", ["Accuracy", "Recall", "F1", "ROC-AUC"], default="Accuracy", key="split_metric")
    pick = pick or "Accuracy"
    with st.container(border=True):
        fig = go.Figure()
        for name in R:
            d = sdf[sdf["Model"] == name]
            fig.add_bar(name=name, x=d["Split (train:test)"], y=d[pick], text=d[pick].map("{:.3f}".format),
                        textposition="outside", marker=dict(color=colors.get(name, "#888"), cornerradius=4))
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
        lo = max(0.0, float(sdf[pick].min()) - 0.06)
        fig.update_yaxes(range=[lo, min(1.0, float(sdf[pick].max()) + 0.04)], gridcolor=ui.pal()["grid"], title=pick)
        st.plotly_chart(ui._layout(fig, 360, title=f"{pick} by split ratio").update_layout(margin=dict(l=10, r=10, t=90, b=10)),
                        width="stretch")
    best_i = sdf.sort_values(["ROC-AUC", "Accuracy"], ascending=False).index[0]
    used_i = sdf.index[(sdf["Split (train:test)"] == MAIN) & (sdf["Model"] == M["model_name"])][0]
    tab = sdf.copy()
    tab["Model"] = tab["Model"].map(label)
    tab["Note"] = ""
    tab.loc[best_i, "Note"] = "Best score"
    tab.loc[used_i, "Note"] = "Used in dashboard" if used_i != best_i else "Best + used in dashboard"

    # one colour per split ratio, ONLY in the first column (light bg / dark text, dark bg / white text)
    _order = list(dict.fromkeys(sdf["Split (train:test)"]))
    _LIGHT = [("#C7D2FE", "#1E1B4B"), ("#A7F3D0", "#064E3B"), ("#FDE68A", "#78350F"), ("#FBCFE8", "#831843")]
    _DARK = [("#3730A3", "#FFFFFF"), ("#0F766E", "#FFFFFF"), ("#B45309", "#FFFFFF"), ("#9D174D", "#FFFFFF")]

    def _split_cell(label):
        bg, fg = (_DARK if ui.dark() else _LIGHT)[_order.index(label) % 4]
        return f"background-color: {bg}; color: {fg}; font-weight: 700"

    def _row(r):
        first = _split_cell(r.iloc[0])
        if r.name == best_i:
            return [first] + ["background-color: #BBF7D0; color: #14532D; font-weight: 700"] * (len(r) - 1)
        if r.name == used_i:
            return [first] + ["background-color: #C7D2FE; color: #1E1B4B; font-weight: 700"] * (len(r) - 1)
        return [first] + [""] * (len(r) - 1)

    ui.table(tab, fmt={"Train rows": "{:,}", "Test rows": "{:,}", "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                       "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}, highlight=_row)
    gap = sdf.loc[best_i, "ROC-AUC"] - sdf.loc[used_i, "ROC-AUC"]
    st.caption(f"Green = best ROC-AUC in this table. Blue = the model the dashboard uses "
               f"({M['model_name']}, {MAIN}). ROC-AUC gap between them: {gap:.3f}.")

    with st.expander("Why 80:20 and not 70:30?"):
        import json
        sn = cfg.ARTIFACT_DIR / "split_noise.json"
        if sn.exists():
            nz = pd.DataFrame(json.loads(sn.read_text())["rows"])
            nz["Accuracy (mean +- sd)"] = nz.apply(lambda r: f"{r.acc_mean:.3f} +- {r.acc_sd:.3f}", axis=1)
            nz["ROC-AUC (mean +- sd)"] = nz.apply(lambda r: f"{r.auc_mean:.3f} +- {r.auc_sd:.3f}", axis=1)
            ui.table(nz[["split", "repeats", "Accuracy (mean +- sd)", "ROC-AUC (mean +- sd)"]].rename(
                columns={"split": "Split", "repeats": "Random splits"}))
            st.caption("Each ratio repeated with different random student splits. If the gap between ratios is smaller "
                       "than the +- spread, the difference is chance, not a real improvement.")
        else:
            st.caption("Run `python split_noise.py` once to show the repeated-split table here.")
        st.markdown("- The proposal fixes an 80:20 student-level split; it is the standard convention.\n"
                    "- A single random split is noisy: the gap between ratios is within split-to-split variation.\n"
                    "- Every ratio has a different test set, so picking the 'best' ratio by test score would select on "
                    "the test data and inflate the result.\n"
                    "- 80:20 keeps the most students for learning while the test set stays large.")

# ------------------------------------------------------------------ training information
st.subheader("Training information")
a, b = st.columns(2)
with a, st.container(border=True):
    st.markdown("**Data**")
    st.markdown(f"- Source: OULAD (7 course modules, 2013-2014)\n"
                f"- Enrolments in the dataset: {D['enrolments_total']:,}; already withdrawn before the early-warning "
                f"point (excluded): {D['already_withdrawn_at_cutoff']:,}\n"
                f"- Eligible enrolments: **{D['enrolments_eligible']:,}** ({D['n_students_unique']:,} unique students)\n"
                f"- Train: **{D['n_train']:,}** enrolments from {D['n_train_students']:,} students\n"
                f"- Test (never seen while learning): **{D['n_test']:,}** enrolments from {D['n_test_students']:,} students\n"
                f"- Target: Withdrawn / Fail = at-risk, Pass / Distinction = not at-risk")
with b, st.container(border=True):
    st.markdown("**Model setup**")
    st.markdown(f"- Prediction point: **{D['weeks_before_exam']} weeks before the final exam**\n"
                f"- Split: **GroupShuffleSplit {MAIN.replace(chr(58), chr(47))} by student** (no student in both sets)\n"
                f"- Imbalance handling: {IMB.get(M['imbalance'], M['imbalance'])}\n"
                f"- Click (VLE) features: **{'used' if M['use_vle'] else 'not used in this build'}**\n"
                f"- Features kept by RFE: **{len(M['selected_columns'])}** columns "
                f"({len(M['selected_units'])} variables)\n"
                f"- Risk levels: Low < {cfg.LOW_MAX:.0%}, Medium {cfg.LOW_MAX:.0%}-{cfg.HIGH_MIN:.0%}, "
                f"High >= {cfg.HIGH_MIN:.0%}\n"
                f"- Best settings: `{best['best_params']}`")

st.subheader("Features selected by RFE")
c1, c2 = st.columns([1, 1.2])
with c1:
    feat = pd.DataFrame({"Variable": [P.units[u]["label"] for u in M["selected_units"]],
                         "Type": [P.units[u]["group"] for u in M["selected_units"]]})
    ui.table(feat)
with c2, st.container(border=True):
    cv = pd.DataFrame(M["rfe_cv"]).sort_values("n_features")
    fig = go.Figure(go.Scatter(x=cv["n_features"], y=cv["auc"], mode="lines+markers",
                               line=dict(color=ui.BRAND, width=3), marker=dict(size=7)))
    fig.add_vline(x=len(M["selected_columns"]), line=dict(dash="dash", color="#E5484D"),
                  annotation_text="chosen", annotation_position="top")
    fig.update_xaxes(title="Number of features", gridcolor=ui.pal()["grid"])
    fig.update_yaxes(title="CV ROC-AUC", gridcolor=ui.pal()["grid"])
    st.plotly_chart(ui._layout(fig, 320, title="RFE: accuracy vs number of features"), width="stretch")

