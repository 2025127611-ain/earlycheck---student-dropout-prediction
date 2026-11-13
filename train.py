"""Train the dropout-risk models on OULAD and save everything the dashboard needs.

    python train.py                  # full run
    python train.py --fast           # small grids, quick test
    python train.py --out some_dir   # write artifacts elsewhere

Pipeline (every learning step uses the TRAIN split only):
  integrate OULAD files -> cutoff 5 weeks before the exam -> target (Withdrawn/Fail = 1, Pass/Distinction = 0)
  -> GroupShuffleSplit 80/20 by id_student -> feature engineering + encoding
  -> RFE (RFECV picks the size, StratifiedGroupKFold) -> SMOTENC on train folds only
  -> grid-search Decision Tree / Random Forest / XGBoost -> evaluate on the untouched test students
  -> best model by cross-validated ROC-AUC -> SHAP (probability space) for the test students
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core import config as cfg
from core import data_prep as dp
from core import explain as ex

warnings.filterwarnings("ignore")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y, p, thr=0.5) -> dict:
    pred = (p >= thr).astype(int)
    alert = (p >= cfg.LOW_MAX).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred), recall=recall_score(y, pred),
                f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), recall_at_alert=recall_score(y, alert),
                precision_at_alert=precision_score(y, alert), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def base_models(fast: bool, spw: float | None, balanced: bool):
    rs = cfg.RANDOM_STATE
    cw = "balanced" if balanced else None
    dt = (DecisionTreeClassifier(random_state=rs, class_weight=cw),
          {"max_depth": [4, 6], "min_samples_leaf": [30]} if fast else
          {"max_depth": [4, 5, 6, 8], "min_samples_leaf": [20, 50], "criterion": ["gini", "entropy"]})
    rf = (RandomForestClassifier(random_state=rs, n_jobs=-1, class_weight=cw),
          {"n_estimators": [150], "max_depth": [8], "min_samples_leaf": [5]} if fast else
          {"n_estimators": [200, 400], "max_depth": [8, 12], "min_samples_leaf": [3, 8], "max_features": ["sqrt"]})
    xgb = (XGBClassifier(random_state=rs, n_jobs=2, eval_metric="logloss", tree_method="hist",
                         scale_pos_weight=spw or 1.0),
           {"n_estimators": [200], "learning_rate": [0.08], "max_depth": [4]} if fast else
           {"n_estimators": [200, 400], "learning_rate": [0.03, 0.08], "max_depth": [3, 5],
            "min_child_weight": [1, 5], "subsample": [0.8], "colsample_bytree": [0.8]})
    return {"Decision Tree": dt, "Random Forest": rf, "XGBoost": xgb}


def split_comparison(table, feats, y, results, best_params_by_model, n_opt, use_vle, imb, fast):
    """Re-run the whole pipeline for 80:20, 70:30 and 60:40 GROUP splits (same models, same tuned settings).
    Encoder and RFE are re-fitted on each split's own training students, so nothing leaks."""
    out = {}
    makers = {"Decision Tree": lambda p: DecisionTreeClassifier(random_state=cfg.RANDOM_STATE, **p),
              "Random Forest": lambda p: RandomForestClassifier(random_state=cfg.RANDOM_STATE, n_jobs=-1, **p),
              "XGBoost": lambda p: XGBClassifier(random_state=cfg.RANDOM_STATE, n_jobs=2, eval_metric="logloss",
                                                 tree_method="hist", **p)}
    for test_size in (0.20, 0.30, 0.40):
        label = f"{round((1 - test_size) * 100)}:{round(test_size * 100)}"
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.RANDOM_STATE)
                      .split(table, y, groups=table["id_student"]))
        assert not set(table.loc[tr, "id_student"]) & set(table.loc[te, "id_student"])
        sp = dp.fit_encoder(feats.loc[tr], use_vle)
        X = dp.encode(feats, sp)
        rf = RFE(RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, n_jobs=-1,
                                        random_state=cfg.RANDOM_STATE), n_features_to_select=n_opt, step=1)
        rf.fit(X.loc[tr], y.loc[tr])
        cols = [c for c, k in zip(X.columns, rf.support_) if k]
        Xtr, Xte = X.loc[tr, cols], X.loc[te, cols]
        cat = [i for i, c in enumerate(cols) if c in sp["binary"] or "=" in c]
        out[label] = {"n_train": int(len(tr)), "n_test": int(len(te)), "models": {}}
        for name, mk in makers.items():
            m = mk(best_params_by_model[name])
            Xfit, yfit = Xtr, y.loc[tr]
            if imb == "smotenc":
                smp = SMOTENC(categorical_features=cat, random_state=cfg.RANDOM_STATE) if cat else SMOTE(random_state=cfg.RANDOM_STATE)
                Xfit, yfit = smp.fit_resample(Xtr, y.loc[tr])
            m.fit(Xfit, yfit)
            mt = metrics(y.loc[te], m.predict_proba(Xte)[:, 1])
            out[label]["models"][name] = {k: float(mt[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
        log(f"split {label}: " + " | ".join(f"{n} acc={v['accuracy']:.3f} auc={v['roc_auc']:.3f}"
                                            for n, v in out[label]["models"].items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(cfg.ARTIFACT_DIR))
    ap.add_argument("--no-vle", action="store_true", help="ignore studentVle.csv even if it exists")
    ap.add_argument("--imbalance", default=cfg.IMBALANCE, choices=["smotenc", "class_weight", "none"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imb = a.imbalance

    # ---------------------------------------------------------------- 1. integrate + clean
    use_vle = dp.has_vle() and not a.no_vle
    table, info = dp.build_table(use_vle)
    qa = dp.quality_checks(table)
    log(f"enrolments: {info['enrolments_total']:,} total | {info['already_withdrawn_at_cutoff']:,} already "
        f"withdrawn before the cutoff (excluded) | {info['enrolments_eligible']:,} eligible | "
        f"{info['n_students_unique']:,} unique students | at-risk share {info['at_risk_share']:.1%}")
    log(f"quality checks: {qa} | VLE features: {'YES' if use_vle else 'NO (studentVle.csv not found)'}")
    table = table.reset_index(drop=True)
    input_cols = [c for c in table.columns if c in dp.INPUT_FIELDS]
    inputs = table[input_cols]
    feats = dp.engineer(inputs)
    y = table["y"].astype(int)

    # ---------------------------------------------------------------- 2. GROUPED split by student
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(table, y, groups=table["id_student"]))
    overlap = set(table.loc[tr_idx, "id_student"]) & set(table.loc[te_idx, "id_student"])
    assert not overlap, "student leakage: the same student is in train and test"
    log(f"GroupShuffleSplit by id_student: train={len(tr_idx):,} rows ({table.loc[tr_idx, 'id_student'].nunique():,} "
        f"students) | test={len(te_idx):,} rows ({table.loc[te_idx, 'id_student'].nunique():,} students) | "
        f"overlapping students = 0 | at-risk train={y[tr_idx].mean():.3f} test={y[te_idx].mean():.3f}")
    g_tr = table.loc[tr_idx, "id_student"].values
    y_tr, y_te = y.loc[tr_idx], y.loc[te_idx]

    spec = dp.fit_encoder(feats.loc[tr_idx], use_vle)
    X_all = dp.encode(feats, spec)
    X_tr, X_te = X_all.loc[tr_idx], X_all.loc[te_idx]
    log(f"{len(spec['columns'])} candidate columns after encoding")

    # ---------------------------------------------------------------- 3. RFE (grouped CV)
    cv = StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    rfe_est = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, n_jobs=-1,
                                     random_state=cfg.RANDOM_STATE, class_weight="balanced" if imb == "class_weight" else None)
    log("RFECV: finding how many features are needed ...")
    rfecv = RFECV(rfe_est, step=cfg.RFE_STEP, cv=cv, scoring="roc_auc", min_features_to_select=cfg.RFE_MIN_FEATURES,
                  n_jobs=1).fit(X_tr, y_tr, groups=g_tr)
    cvr = rfecv.cv_results_
    res = pd.DataFrame({"n_features": np.asarray(cvr["n_features"]).ravel().astype(int),
                        "mean_test_score": np.asarray(cvr["mean_test_score"]).ravel()})
    ok = res[res["n_features"] <= cfg.RFE_MAX_FEATURES]
    best = ok["mean_test_score"].max()
    n_opt = int(ok.loc[ok["mean_test_score"] >= best - cfg.RFE_TOLERANCE, "n_features"].min())
    log(f"best CV AUC within {cfg.RFE_MIN_FEATURES}-{cfg.RFE_MAX_FEATURES} features = {best:.4f} "
        f"(all features: {res['mean_test_score'].max():.4f}); chosen size: {n_opt}")
    rfe = RFE(rfe_est, n_features_to_select=n_opt, step=1).fit(X_tr, y_tr)
    selected = [c for c, s in zip(X_tr.columns, rfe.support_) if s]
    ranking = pd.Series(rfe.ranking_, index=X_tr.columns).sort_values()
    selected_units = sorted({dp.column_to_unit(c) for c in selected})
    log("RFE selected columns: " + ", ".join(selected))
    Xs_tr, Xs_te, Xs_all = X_tr[selected], X_te[selected], X_all[selected]

    # ---------------------------------------------------------------- 4. tune + evaluate
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum()) if imb == "class_weight" else None
    cat_idx = [i for i, c in enumerate(selected) if c in spec["binary"] or "=" in c]
    results, fitted = {}, {}
    for name, (est, grid) in base_models(a.fast, spw, imb == "class_weight").items():
        t0 = time.time()
        if imb == "smotenc":
            sampler = (SMOTENC(categorical_features=cat_idx, random_state=cfg.RANDOM_STATE) if cat_idx
                       else SMOTE(random_state=cfg.RANDOM_STATE))
            est = ImbPipeline([("smote", sampler), ("model", est)])          # oversample TRAIN folds only
            grid = {f"model__{k}": v for k, v in grid.items()}
        gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True).fit(Xs_tr, y_tr, groups=g_tr)
        mdl = gs.best_estimator_.named_steps["model"] if imb == "smotenc" else gs.best_estimator_
        m = metrics(y_te, mdl.predict_proba(Xs_te)[:, 1])
        params = {k.replace("model__", ""): (v if v is None or isinstance(v, (int, float, str)) else str(v))
                  for k, v in gs.best_params_.items()}
        m.update(cv_auc=float(gs.best_score_), best_params=params)
        results[name], fitted[name] = m, mdl
        log(f"{name:14s} CV-AUC={m['cv_auc']:.4f} | test acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']:.3f} brier={m['brier']:.3f} | "
            f"recall@{cfg.LOW_MAX:.2f}={m['recall_at_alert']:.3f} ({time.time() - t0:.0f}s) {params}")
    best_name = max(results, key=lambda k: results[k]["cv_auc"])
    model = fitted[best_name]
    log(f"SELECTED MODEL: {best_name} (highest cross-validated ROC-AUC on the training set)")

    # ---------------------------------------------------------------- 5. scores + SHAP (test students)
    prob_all = pd.Series(model.predict_proba(Xs_all)[:, 1], index=table.index)
    bg = Xs_tr.sample(min(cfg.SHAP_BACKGROUND_SIZE, len(Xs_tr)), random_state=cfg.RANDOM_STATE)
    explainer = ex.build_explainer(model, bg)
    t0 = time.time()
    sv = ex.shap_values(explainer, Xs_te, check=False)
    base = ex.base_value(explainer)
    gap = float(np.abs(base + sv.sum(1) - prob_all.loc[te_idx].values).max())
    log(f"SHAP for {len(te_idx):,} test students in {time.time() - t0:.0f}s | base risk={base:.3f} | max additivity gap={gap:.2e}")
    assert gap < 2e-2, "SHAP is not additive to the predicted probability"

    log("Comparing 80:20, 70:30 and 60:40 grouped splits ...")
    bp = {n: {k: v for k, v in results[n]["best_params"].items()} for n in results}
    split_cmp = split_comparison(table, feats, y, results, bp, n_opt, use_vle, imb, a.fast)

    # ---------------------------------------------------------------- 6. save
    split = pd.Series("Train", index=table.index)
    split[te_idx] = "Test"
    students = pd.DataFrame({"student_id": table["student_id"], "split": split.values, "actual": table["final_result"],
                             "risk": prob_all.values, "level": [ex.risk_level(p) for p in prob_all],
                             "code_module": table["code_module"], "code_presentation": table["code_presentation"]})
    students.to_csv(out / "students.csv", index=False)
    required = dp.required_inputs(selected_units, use_vle)
    te_ids = table.loc[te_idx, "student_id"].values
    ti = inputs.loc[te_idx, required].copy()
    ti.index = te_ids
    ti.index.name = "student_id"
    ti.to_csv(out / "test_inputs.csv")
    tf = feats.loc[te_idx, selected_units].copy()
    tf.index = te_ids
    tf.index.name = "student_id"
    tf.to_csv(out / "test_feats.csv")
    np.save(out / "test_shap.npy", sv.astype(np.float32))
    bg.to_csv(out / "background.csv", index=False)
    joblib.dump(model, out / "model.joblib", compress=3)
    joblib.dump(spec, out / "encoder.joblib")
    cut = dp.cutoff_table()
    data_info = dict(info, quality=qa, n_train=int(len(tr_idx)), n_test=int(len(te_idx)),
                     n_train_students=int(table.loc[tr_idx, "id_student"].nunique()),
                     n_test_students=int(table.loc[te_idx, "id_student"].nunique()),
                     weeks_before_exam=cfg.WEEKS_BEFORE_EXAM, cutoffs=cut.set_index(["code_module", "code_presentation"])["cutoff"]
                     .rename(lambda x: x).reset_index().assign(k=lambda d: d.code_module + " " + d.code_presentation)
                     .set_index("k")["cutoff"].astype(int).to_dict())
    meta = dict(model_name=best_name, use_vle=use_vle, selected_columns=selected, selected_units=selected_units,
                required_inputs=required, base_value=base, thresholds=dict(low_max=cfg.LOW_MAX, high_min=cfg.HIGH_MIN),
                imbalance=imb, results=results, data_info=data_info, rfe_ranking={k: int(v) for k, v in ranking.items()},
                rfe_cv=[{"n_features": int(r.n_features), "auc": float(r.mean_test_score)} for r in res.itertuples()],
                exclude_sensitive=cfg.EXCLUDE_SENSITIVE, split_comparison=split_cmp)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    log(f"saved artifacts to {out}")
    log("Inputs the live form will ask for: " + ", ".join(required))


if __name__ == "__main__":
    main()
