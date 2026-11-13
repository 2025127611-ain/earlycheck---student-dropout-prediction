"""OULAD: integrate the CSV files, build ONE row per student-enrolment, engineer features, encode.

Unit of analysis = one enrolment (code_module + code_presentation + id_student).  A student can appear in
several enrolments, so the train/test split is GROUPED by id_student (GroupShuffleSplit).

All behavioural features use only information up to the early-warning CUTOFF day:

    exam day (or presentation length)  -  WEEKS_BEFORE_EXAM * 7

    studentInfo + studentRegistration + courses/assessments + studentAssessment + studentVle/vle
        --build_table-->  INPUTS (plain numbers / labels)  --engineer-->  FEATURES  --encode-->  X
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

# --------------------------------------------------------------------------- label orders
EDU_LEVELS = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification",
              "Post Graduate Qualification"]
AGE_LEVELS = ["0-35", "35-55", "55<="]
IMD_LEVELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%",
              "90-100%"]
IMD_UNKNOWN = "Unknown"
REGIONS = ["East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region",
           "North Western Region", "Scotland", "South East Region", "South Region", "South West Region",
           "Wales", "West Midlands Region", "Yorkshire Region"]
MODULES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
PRESENTATIONS = ["2013B", "2013J", "2014B", "2014J"]
ALL_VLE_TYPES = cfg.VLE_TYPES + ["other"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- loading
def _read(name: str, **kw) -> pd.DataFrame:
    p = cfg.DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Put the OULAD CSV files in the data/ folder.")
    return pd.read_csv(p, **kw)


def has_vle() -> bool:
    return cfg.VLE_FILE.exists()


def cutoff_table() -> pd.DataFrame:
    """One row per module-presentation with the early-warning cutoff day."""
    courses = _read("courses.csv")
    ass = _read("assessments.csv")
    exam = (ass[ass["assessment_type"] == "Exam"].groupby(["code_module", "code_presentation"])["date"]
            .max().rename("exam_date").reset_index())
    t = courses.merge(exam, how="left", on=["code_module", "code_presentation"])
    t["exam_day"] = t["exam_date"].fillna(t["module_presentation_length"])
    t["cutoff"] = (t["exam_day"] - cfg.WEEKS_BEFORE_EXAM * 7).astype(int)
    return t[["code_module", "code_presentation", "module_presentation_length", "exam_day", "cutoff"]]


# --------------------------------------------------------------------------- assessment features
def assessment_features(cut: pd.DataFrame) -> pd.DataFrame:
    """Per enrolment: assessments due by the cutoff, submitted, average score, late submissions."""
    ass = _read("assessments.csv")
    ass = ass[(ass["assessment_type"] != "Exam") & ass["date"].notna()]
    ass = ass.merge(cut[["code_module", "code_presentation", "cutoff"]], on=["code_module", "code_presentation"])
    due = ass[ass["date"] <= ass["cutoff"]]
    n_due = (due.groupby(["code_module", "code_presentation"]).size().rename("assess_due").reset_index())

    sa = _read("studentAssessment.csv")
    sa = sa.merge(due[["id_assessment", "code_module", "code_presentation", "date", "cutoff"]], on="id_assessment")
    sa = sa[sa["date_submitted"] <= sa["cutoff"]]                       # submitted by the cutoff
    sa["late"] = (sa["date_submitted"] > sa["date"]).astype(int)
    g = sa.groupby(["code_module", "code_presentation", "id_student"])
    feat = pd.DataFrame({"assess_submitted": g.size(), "avg_score": g["score"].mean(),
                         "n_late": g["late"].sum()}).reset_index()
    return n_due, feat


# --------------------------------------------------------------------------- VLE features (chunked)
def vle_features(enroll_keys: pd.DataFrame, cut: pd.DataFrame, chunk_rows: int = cfg.VLE_CHUNK_ROWS) -> pd.DataFrame:
    """Aggregate studentVle.csv (10M+ rows) in chunks, keeping only clicks on/before the cutoff day.

    Returns one row per enrolment: total_clicks, active_days, n_sites, clicks_recent, clk_<type>.
    """
    mod_i = {m: i for i, m in enumerate(MODULES)}
    pres_i = {p: i for i, p in enumerate(PRESENTATIONS)}
    cut_arr = np.full(len(MODULES) * len(PRESENTATIONS), -10 ** 6, dtype=np.int32)
    for r in cut.itertuples():
        cut_arr[mod_i[r.code_module] * 4 + pres_i[r.code_presentation]] = r.cutoff

    sites = _read("vle.csv", usecols=["id_site", "activity_type"])
    stype = sites.set_index("id_site")["activity_type"].where(sites.set_index("id_site")["activity_type"].isin(cfg.VLE_TYPES),
                                                              "other")
    type_codes = {t: i for i, t in enumerate(ALL_VLE_TYPES)}
    site_code = stype.map(type_codes)

    tot, rec, bytype, days, usites = [], [], [], [], []
    n = 0
    t0 = time.time()
    dtypes = {"id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"}
    for ch in pd.read_csv(cfg.VLE_FILE, usecols=["code_module", "code_presentation", "id_student", "id_site",
                                                  "date", "sum_click"], dtype={**dtypes, "code_module": "category",
                                                                              "code_presentation": "category"},
                          chunksize=chunk_rows):
        n += len(ch)
        mi = ch["code_module"].astype(str).map(mod_i).astype("int32").values
        pi = ch["code_presentation"].astype(str).map(pres_i).astype("int32").values
        cutoff = cut_arr[mi * 4 + pi]
        keep = ch["date"].values <= cutoff
        ch, mi, pi, cutoff = ch[keep], mi[keep], pi[keep], cutoff[keep]
        key = ch["id_student"].values.astype("int64") * 100 + mi * 10 + pi
        d = pd.DataFrame({"key": key, "date": ch["date"].values, "site": ch["id_site"].values,
                          "clk": ch["sum_click"].values, "recent": ch["date"].values > (cutoff - cfg.RECENT_DAYS),
                          "tp": ch["id_site"].map(site_code).fillna(type_codes["other"]).astype("int8").values})
        tot.append(d.groupby("key")["clk"].sum())
        rec.append(d[d["recent"]].groupby("key")["clk"].sum())
        bytype.append(d.groupby(["key", "tp"])["clk"].sum())
        days.append(d[["key", "date"]].drop_duplicates())
        usites.append(d[["key", "site"]].drop_duplicates())
        log(f"  VLE rows read: {n:,}  ({time.time() - t0:.0f}s)")

    out = pd.DataFrame({"total_clicks": pd.concat(tot).groupby(level=0).sum()})
    out["clicks_recent"] = pd.concat(rec).groupby(level=0).sum()
    out["active_days"] = pd.concat(days).drop_duplicates().groupby("key").size()
    out["n_sites"] = pd.concat(usites).drop_duplicates().groupby("key").size()
    bt = pd.concat(bytype).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    bt.columns = [f"clk_{ALL_VLE_TYPES[c]}" for c in bt.columns]
    out = out.join(bt, how="left")
    for t in ALL_VLE_TYPES:
        if f"clk_{t}" not in out:
            out[f"clk_{t}"] = 0
    out = out.fillna(0)
    out["id_student"] = (out.index // 100).astype("int64")
    out["code_module"] = [MODULES[i] for i in (out.index % 100) // 10]
    out["code_presentation"] = [PRESENTATIONS[i] for i in out.index % 10]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- the modelling table
def build_table(use_vle: bool | None = None) -> tuple[pd.DataFrame, dict]:
    """Integrate all OULAD files. Returns (table, info). Columns are the live-form INPUT names + ids + target."""
    use_vle = has_vle() if use_vle is None else use_vle
    info: dict = {"use_vle": bool(use_vle)}
    cut = cutoff_table()
    si = _read("studentInfo.csv")
    reg = _read("studentRegistration.csv")
    info["enrolments_total"] = len(si)

    # ---- data cleaning ----------------------------------------------------------------
    keys = ["code_module", "code_presentation", "id_student"]
    info["duplicate_rows"] = int(si.duplicated(keys).sum())
    si = si.drop_duplicates(keys)
    reg = reg.drop_duplicates(keys)
    info["missing_imd"] = int(si["imd_band"].isna().sum())
    si["imd_band"] = si["imd_band"].fillna(IMD_UNKNOWN).replace({"10-20": "10-20%"})       # fix inconsistent label
    info["missing_registration_date"] = int(reg["date_registration"].isna().sum())

    t = si.merge(reg, on=keys, how="left").merge(cut, on=["code_module", "code_presentation"], how="left")

    # ---- eligibility: still registered at the cutoff ----------------------------------
    gone = t["date_unregistration"].notna() & (t["date_unregistration"] <= t["cutoff"])
    info["already_withdrawn_at_cutoff"] = int(gone.sum())
    if cfg.EXCLUDE_ALREADY_WITHDRAWN:
        t = t[~gone]
    info["enrolments_eligible"] = len(t)

    # ---- target (final_result is only used here, never as a feature) --------------------
    t["y"] = t["final_result"].isin(["Withdrawn", "Fail"]).astype(int)
    t = t.rename(columns={"date_registration": "reg_timing"})

    # ---- assessment features ----------------------------------------------------------
    n_due, af = assessment_features(cut)
    t = t.merge(n_due, on=["code_module", "code_presentation"], how="left").merge(af, on=keys, how="left")
    t["assess_due"] = t["assess_due"].fillna(0).astype(int)
    t["assess_submitted"] = t["assess_submitted"].fillna(0).astype(int).clip(upper=t["assess_due"])
    t["n_late"] = t["n_late"].fillna(0).astype(int)
    t["avg_score"] = t["avg_score"].fillna(0.0)           # no submission -> 0 (flagged by 'no_submission')

    # ---- VLE features -----------------------------------------------------------------
    vle_cols: list[str] = []
    if use_vle:
        log("aggregating studentVle.csv in chunks (this is the slow step) ...")
        vf = vle_features(t[keys], cut)
        t = t.merge(vf, on=keys, how="left")
        vle_cols = ["total_clicks", "clicks_recent", "active_days", "n_sites"] + [f"clk_{x}" for x in ALL_VLE_TYPES]
        t[vle_cols] = t[vle_cols].fillna(0)
        t["clicks_recent"] = t["clicks_recent"].clip(upper=t["total_clicks"])
    else:
        log("WARNING: data/studentVle.csv not found -> VLE click features are NOT used.")

    t = t.reset_index(drop=True)
    t["student_id"] = (t["id_student"].astype(str) + "_" + t["code_module"] + "_" + t["code_presentation"])
    t["region"] = t["region"].astype(str)
    t["gender"] = (t["gender"] == "M").astype(int)
    t["disability"] = (t["disability"] == "Y").astype(int)
    info["at_risk_share"] = float(t["y"].mean())
    info["n_students_unique"] = int(t["id_student"].nunique())
    cols = (["student_id", "id_student", "final_result", "y", "gender", "disability", "region", "highest_education",
             "imd_band", "age_band", "num_of_prev_attempts", "studied_credits", "code_module", "code_presentation",
             "reg_timing", "assess_due", "assess_submitted", "avg_score", "n_late"] + vle_cols)
    info["vle_cols"] = vle_cols
    return t[cols], info


def quality_checks(t: pd.DataFrame) -> dict:
    """Out-of-range / impossible value checks after integration."""
    return {"duplicated_enrolments": int(t.duplicated(["id_student", "code_module", "code_presentation"]).sum()),
            "submitted_gt_due": int((t["assess_submitted"] > t["assess_due"]).sum()),
            "avg_score_out_of_range": int(((t["avg_score"] < 0) | (t["avg_score"] > 100)).sum()),
            "missing_values": int(t.isna().sum().sum())}


# --------------------------------------------------------------------------- variable registry
def U(key, label, group, kind, raw, fmt, **kw):
    return dict(key=key, label=label, group=group, kind=kind, raw=raw, fmt=fmt, **kw)


def candidate_units(use_vle: bool) -> list[dict]:
    units = [
        U("gender", "Gender (male)", "Identity", "binary", ["gender"], "yesno"),
        U("disability", "Disability", "Identity", "binary", ["disability"], "yesno"),
        U("region", "Region", "Identity", "category", ["region"], "text", levels=REGIONS),
        U("highest_education", "Highest education", "Background", "numeric", ["highest_education"], "ordinal",
          levels=EDU_LEVELS),
        U("imd_band", "Area deprivation (IMD band, 0-10% = most deprived)", "Background", "numeric",
          ["imd_band"], "ordinal", levels=IMD_LEVELS),
        U("age_band", "Age band", "Background", "numeric", ["age_band"], "ordinal", levels=AGE_LEVELS),
        U("num_of_prev_attempts", "Previous attempts at this module", "Background", "numeric",
          ["num_of_prev_attempts"], "int"),
        U("studied_credits", "Credits studied", "Course load", "numeric", ["studied_credits"], "int"),
        U("code_module", "Module", "Module", "category", ["code_module"], "text", levels=MODULES),
        U("code_presentation", "Presentation (intake)", "Module", "category", ["code_presentation"], "text",
          levels=PRESENTATIONS),
        U("reg_timing", "Registration day (negative = days before course start)", "Registration", "numeric",
          ["reg_timing"], "reg"),
        # assessments
        U("assess_due", "Assessments due so far", "Assessment", "numeric", ["assess_due"], "int"),
        U("assess_submitted", "Assessments submitted", "Assessment", "numeric", ["assess_submitted"], "int"),
        U("n_missing", "Missing assessments (due but not submitted)", "Assessment", "numeric",
          ["assess_due", "assess_submitted"], "int"),
        U("missing_rate", "Share of due assessments missing", "Assessment", "numeric",
          ["assess_due", "assess_submitted"], "pct"),
        U("n_late", "Late submissions", "Assessment", "numeric", ["n_late"], "int"),
        U("late_rate", "Share of submissions that were late", "Assessment", "numeric",
          ["n_late", "assess_submitted"], "pct"),
        U("avg_score", "Average assessment score (0-100)", "Academic performance", "numeric", ["avg_score"],
          "grade"),
        U("no_submission", "No assessment submitted yet", "Assessment", "binary", ["assess_submitted"], "yesno"),
    ]
    if use_vle:
        units += [
            U("total_clicks", "Total VLE clicks", "VLE engagement", "numeric", ["total_clicks"], "int"),
            U("active_days", "Days active on the VLE", "VLE engagement", "numeric", ["active_days"], "int"),
            U("n_sites", "Different VLE materials opened", "VLE engagement", "numeric", ["n_sites"], "int"),
            U("clicks_recent", "VLE clicks in the last 4 weeks", "VLE engagement", "numeric", ["clicks_recent"],
              "int"),
            U("clicks_per_active_day", "Average clicks per active day", "VLE engagement", "numeric",
              ["total_clicks", "active_days"], "ratio"),
            U("recent_share", "Share of clicks in the last 4 weeks", "VLE engagement", "numeric",
              ["total_clicks", "clicks_recent"], "pct"),
            U("no_vle", "No VLE activity at all", "VLE engagement", "binary", ["total_clicks"], "yesno"),
        ]
        for t in ALL_VLE_TYPES:
            units.append(U(f"clk_{t}", f"VLE clicks: {t}", "VLE engagement", "numeric", [f"clk_{t}"], "int"))
    if cfg.EXCLUDE_SENSITIVE:
        units = [u for u in units if u["key"] not in cfg.SENSITIVE_COLUMNS]
    return units


def unit_map(use_vle: bool) -> dict[str, dict]:
    return {u["key"]: u for u in candidate_units(use_vle)}


def required_inputs(selected_units: list[str], use_vle: bool) -> list[str]:
    um = unit_map(use_vle)
    need: set[str] = set()
    for u in selected_units:
        need.update(um[u]["raw"])
    return sorted(need)


# --------------------------------------------------------------------------- live-form fields
def _int(label, mx, default, help_=""):
    return dict(type="int", label=label, min=0, max=mx, default=default, help=help_)


INPUT_FIELDS: dict[str, dict] = {
    "gender": dict(type="binary", label="Male"),
    "disability": dict(type="binary", label="Has a declared disability"),
    "region": dict(type="select", label="Region", options=REGIONS),
    "highest_education": dict(type="select", label="Highest education", options=EDU_LEVELS,
                              default="A Level or Equivalent"),
    "imd_band": dict(type="select", label="Area deprivation (IMD band)", options=IMD_LEVELS + [IMD_UNKNOWN],
                     default="50-60%", help="0-10% = most deprived area, 90-100% = least deprived."),
    "age_band": dict(type="select", label="Age band", options=AGE_LEVELS),
    "num_of_prev_attempts": _int("Previous attempts at this module", 6, 0),
    "studied_credits": dict(type="int", label="Credits studied", min=30, max=655, default=60),
    "code_module": dict(type="select", label="Module", options=MODULES),
    "code_presentation": dict(type="select", label="Presentation (intake)", options=PRESENTATIONS,
                              default="2014J"),
    "reg_timing": dict(type="int", label="Registration day (negative = days before course start)", min=-322, max=167,
                       default=-60, help="Negative = registered that many days BEFORE the course started."),
    "assess_due": _int("Assessments due so far", 20, 4, "Number of TMA/CMA assessments with a deadline up to "
                                                         "the prediction point (about 5 weeks before the exam)."),
    "assess_submitted": _int("Assessments submitted", 20, 4, "How many of those the student handed in. "
                                                             "Cannot exceed assessments due."),
    "avg_score": dict(type="float", label="Average assessment score (0-100)", min=0.0, max=100.0, default=70.0,
                      help="Average of submitted assessments. Use 0 if nothing was submitted."),
    "n_late": _int("Late submissions", 20, 0, "Submissions handed in after the deadline. "
                                              "Cannot exceed assessments submitted."),
    "total_clicks": dict(type="int", label="Total VLE clicks", min=0, max=60000, default=800,
                         help="All clicks on the learning platform up to the prediction point."),
    "active_days": _int("Days active on the VLE", 330, 40, "Number of different days with at least one click."),
    "n_sites": _int("Different VLE materials opened", 1200, 60),
    "clicks_recent": dict(type="int", label="VLE clicks in the last 4 weeks", min=0, max=30000, default=150,
                          help="Clicks during the 4 weeks before the prediction point."),
}
for _t in ALL_VLE_TYPES:
    INPUT_FIELDS[f"clk_{_t}"] = dict(type="int", label=f"VLE clicks: {_t}", min=0, max=40000, default=50)

INPUT_SECTIONS = {
    "Student information": ["code_module", "code_presentation", "highest_education", "imd_band", "age_band",
                            "num_of_prev_attempts", "studied_credits", "gender", "disability", "region"],
    "Registration": ["reg_timing"],
    "Assessments": ["assess_due", "assess_submitted", "avg_score", "n_late"],
    "VLE activity": ["total_clicks", "active_days", "n_sites", "clicks_recent"]
                                                 + [f"clk_{t}" for t in ALL_VLE_TYPES],
}


# --------------------------------------------------------------------------- INPUTS -> FEATURES
def _ord(s: pd.Series, levels: list[str]) -> pd.Series:
    return s.map({v: i for i, v in enumerate(levels)})


def _safe_div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def engineer(inp: pd.DataFrame) -> pd.DataFrame:
    """Derived features (only where their inputs exist). Ordinal labels become numbers; categories stay text."""
    f = inp.copy()
    has = lambda *c: all(x in f for x in c)  # noqa: E731
    if "highest_education" in f:
        f["highest_education"] = _ord(f["highest_education"], EDU_LEVELS)
    if "age_band" in f:
        f["age_band"] = _ord(f["age_band"], AGE_LEVELS)
    if "imd_band" in f:
        f["imd_band"] = _ord(f["imd_band"], IMD_LEVELS)             # 'Unknown' -> NaN (filled when encoding)
    if has("assess_due", "assess_submitted"):
        f["n_missing"] = (f["assess_due"] - f["assess_submitted"]).clip(lower=0)
        f["missing_rate"] = _safe_div(f["n_missing"], f["assess_due"])
    if has("assess_submitted"):
        f["no_submission"] = (f["assess_submitted"] == 0).astype(int)
    if has("n_late", "assess_submitted"):
        f["late_rate"] = _safe_div(f["n_late"], f["assess_submitted"])
    if has("total_clicks", "active_days"):
        f["clicks_per_active_day"] = _safe_div(f["total_clicks"], f["active_days"])
    if has("total_clicks", "clicks_recent"):
        f["recent_share"] = _safe_div(f["clicks_recent"], f["total_clicks"])
    if has("total_clicks"):
        f["no_vle"] = (f["total_clicks"] == 0).astype(int)
    return f


# --------------------------------------------------------------------------- FEATURES -> X
def fit_encoder(feat_train: pd.DataFrame, use_vle: bool, min_share: float = 0.0) -> dict:
    """Learn on the TRAIN split only: missing-value fills and the one-hot column list."""
    spec = {"use_vle": use_vle, "numeric": [], "binary": [], "category": {}, "fill": {}, "columns": []}
    for u in candidate_units(use_vle):
        k = u["key"]
        if u["kind"] == "numeric":
            spec["numeric"].append(k)
            if feat_train[k].isna().any():
                spec["fill"][k] = float(feat_train[k].median())
        elif u["kind"] == "binary":
            spec["binary"].append(k)
        else:
            spec["category"][k] = {"kept": sorted(feat_train[k].dropna().unique().tolist()), "other": False}
    cols = spec["numeric"] + spec["binary"]
    for k, c in spec["category"].items():
        cols += [f"{k}={lv}" for lv in c["kept"]]
    spec["columns"] = cols
    return spec


def encode(feat: pd.DataFrame, spec: dict, only: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode with frozen train categories; fixed column order. `only` = just these columns."""
    want = set(only) if only is not None else None
    X = pd.DataFrame(index=feat.index)
    for k in spec["numeric"] + spec["binary"]:
        if want is None or k in want:
            X[k] = pd.to_numeric(feat[k]).astype(float).fillna(spec["fill"].get(k, 0.0))
    for k, c in spec["category"].items():
        levels = c["kept"]
        if want is not None and not any(f"{k}={lv}" in want for lv in levels):
            continue
        for lv in levels:
            if want is None or f"{k}={lv}" in want:
                X[f"{k}={lv}"] = (feat[k] == lv).astype(float)
    return X[only if only is not None else spec["columns"]]


def column_to_unit(col: str) -> str:
    return col.split("=")[0]
