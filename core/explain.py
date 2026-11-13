"""SHAP helpers, plain-language explanation text and intervention mapping.

SHAP values are computed IN PROBABILITY SPACE (shap.TreeExplainer, interventional,
model_output='probability'), so for every student:

    base risk  +  sum(contributions)  =  predicted risk          (exactly)

Contributions are shown in "percentage points" of risk.  Dummy columns of the same
variable (e.g. course=Nursing, course=Other) are added together so each variable
appears once, with a readable name and the student's actual value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .data_prep import column_to_unit

# --------------------------------------------------------------------------- risk tiers
def risk_level(p: float) -> str:
    if p >= cfg.HIGH_MIN:
        return "High"
    if p >= cfg.LOW_MAX:
        return "Medium"
    return "Low"


def pct(p: float) -> str:
    """Show risk as a whole percent but never claim certainty: clamp to 1%-99%."""
    return f"{min(max(p * 100, 1), 99):.0f}%"


RISK_COLOR = {"High": "#E5484D", "Medium": "#F5A524", "Low": "#2FB380"}
RISK_TEXT = {"High": "#FFFFFF", "Medium": "#3B2A00", "Low": "#FFFFFF"}
UP_COLOR = "#E5484D"       # pushes risk up   (warm)
DOWN_COLOR = "#3B82F6"     # pushes risk down (cool)
BRAND = "#4F46E5"


# --------------------------------------------------------------------------- SHAP
def build_explainer(model, background: pd.DataFrame):
    import shap
    return shap.TreeExplainer(model, data=background, feature_perturbation="interventional",
                              model_output="probability")


def shap_values(explainer, X: pd.DataFrame, check: bool = False) -> np.ndarray:
    """(n_rows, n_features) contributions to P(dropout), in probability units."""
    sv = explainer.shap_values(X, check_additivity=check)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def base_value(explainer) -> float:
    ev = np.asarray(explainer.expected_value).ravel()
    return float(ev[-1])


def aggregate_units(sv: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Add dummy columns of the same variable together -> one column per variable."""
    df = pd.DataFrame(sv, columns=columns)
    return df.T.groupby(column_to_unit).sum().T


# --------------------------------------------------------------------------- formatting
def fmt_value(unit: dict, v) -> str:
    f = unit["fmt"]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Unknown"
    if f == "yesno":
        return "Yes" if float(v) >= 0.5 else "No"
    if f == "pct":
        return f"{float(v) * 100:.0f}%"
    if f == "int":
        return f"{int(round(float(v)))}"
    if f == "ratio":
        return f"{float(v):.1f}"
    if f == "grade":
        return f"{float(v):.1f}"
    if f == "reg":
        d = int(round(float(v)))
        return f"{abs(d)} days {'before' if d <= 0 else 'after'} start"
    if f == "ordinal":
        lv = unit["levels"]
        return lv[min(max(int(round(float(v))), 0), len(lv) - 1)]
    return str(v)


def local_table(contrib_units: pd.Series, feat_row: pd.Series, um: dict) -> pd.DataFrame:
    """One row per variable: name, student's value, contribution (percentage points)."""
    rows = []
    for k, c in contrib_units.items():
        u = um[k]
        rows.append(dict(key=k, feature=u["label"], group=u["group"], value=fmt_value(u, feat_row[k]),
                         points=float(c) * 100))
    df = pd.DataFrame(rows)
    return df.reindex(df["points"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- plain-language text
def narrative(prob: float, base: float, tbl: pd.DataFrame, top: int = 3) -> str:
    lvl = risk_level(prob)
    ups = tbl[tbl.points > 0.3].head(top)
    downs = tbl[tbl.points < -0.3].sort_values("points").head(top)
    s = ""
    if len(ups):
        s += "**Biggest reasons the risk is higher (red bars):**\n"
        for _, r in ups.iterrows():
            s += f"- {r.feature} = **{r.value}** (+{r.points:.1f} points)\n"
    if len(downs):
        s += "\n**Reasons the risk is lower (blue bars):**\n"
        for _, r in downs.iterrows():
            s += f"- {r.feature} = **{r.value}** ({r.points:.1f} points)\n"
    return s


# --------------------------------------------------------------------------- interventions (Table 3.5)
_ACTIONS = {
    "VLE engagement": "Encourage engagement with the learning activities and send an academic feedback reminder.",
    "Assessment": "Send an assessment reminder (missing / late work) and arrange lecturer follow-up.",
    "Academic performance": "Recommend academic support: tutoring, study-skills sessions or a lecturer consultation.",
    "Registration": "Check in about late registration: orientation and a study-plan session.",
    "Course load": "Academic advising on workload: check the credits studied against the time available.",
    "Module": "Advisor check-in about the module's workload and expectations.",
}
_NON_ACTIONABLE = {"Identity", "Background"}


def recommend(tbl: pd.DataFrame, level: str, top: int = 3) -> list[str]:
    """Intervention text driven by the main risk-increasing SHAP factors AND the risk level."""
    ups = tbl[(tbl.points > 0.5) & (~tbl.group.isin(_NON_ACTIONABLE))].head(top)
    out, seen = [], set()
    for _, r in ups.iterrows():
        if r.group in seen or r.group not in _ACTIONS:
            continue
        seen.add(r.group)
        out.append(f"**{r.group}** ({r.feature}): {_ACTIONS[r.group]}")
    if level == "High" and len(ups) >= 2:
        out.append("**Multiple risk factors + high risk:** proactive lecturer / advisor follow-up, "
                   "ideally by direct message (WhatsApp).")
    if not out:
        out.append("No strong actionable risk factor detected. Keep monitoring through the semester."
                   if level == "Low" else
                   "Risk is driven mainly by background factors. Offer a general advisor check-in.")
    return out


_SHORT = {"Academic performance": "Academic support / tutoring", "Assessment": "Assessment reminder + lecturer follow-up",
          "Course load": "Course-load advising", "Finance": "Finance / scholarship referral",
          "Entry profile": "Mentoring / bridging programme", "Programme": "Advisor check-in (programme fit)"}


def short_action(level: str, group: str) -> str:
    """One-line suggested action for the student table."""
    if level == "Low":
        return "Monitor"
    return _SHORT.get(group, "Advisor check-in")


def monitoring_plan(tbl: pd.DataFrame) -> list[str]:
    """Low-risk students get a monitoring list instead of a message."""
    items = ["Continue routine monitoring through the semester.",
             "Re-check the risk score after the next set of assessment results.",
             "Remind the student that tutoring and advisor support are available any time."]
    watch = tbl[(tbl.points > 0.5) & (~tbl.group.isin(_NON_ACTIONABLE))].head(2)
    for r in watch.itertuples():
        items.append(f"Keep an eye on: {r.feature} ({r.value}).")
    return items


def suggested_action(level: str) -> str:
    return {"High": "Priority support", "Medium": "Advisor check-in", "Low": "Routine monitoring"}[level]
