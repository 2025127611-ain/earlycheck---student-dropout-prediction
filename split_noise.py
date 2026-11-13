"""Is 70:30 really better than 80:20, or is it just luck of one random split?

Repeats each train/test ratio with several different random GROUP splits (same student never in train and test)
and reports mean +- standard deviation of the test scores. Uses the selected model's tuned settings from
artifacts/meta.json. Writes artifacts/split_noise.json, which the Model Performance page shows automatically.

    python split_noise.py            # 5 repeats per ratio
    python split_noise.py --repeats 10
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core import config as cfg
from core import data_prep as dp

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=5)
a = ap.parse_args()

meta = json.loads((cfg.ARTIFACT_DIR / "meta.json").read_text())
name = meta["model_name"]
params = meta["results"][name]["best_params"]
n_opt = len(meta["selected_columns"])
use_vle = meta["use_vle"]
make = {"Decision Tree": lambda: DecisionTreeClassifier(random_state=cfg.RANDOM_STATE, **params),
        "Random Forest": lambda: RandomForestClassifier(random_state=cfg.RANDOM_STATE, n_jobs=-1, **params),
        "XGBoost": lambda: XGBClassifier(random_state=cfg.RANDOM_STATE, n_jobs=2, eval_metric="logloss",
                                         tree_method="hist", **params)}[name]

table, _ = dp.build_table(use_vle)
table = table.reset_index(drop=True)
feats = dp.engineer(table[[c for c in table.columns if c in dp.INPUT_FIELDS]])
y = table["y"].astype(int)
rows = []
for ts in (0.20, 0.30, 0.40):
    label = f"{round((1 - ts) * 100)}:{round(ts * 100)}"
    acc, auc = [], []
    for r in range(a.repeats):
        t0 = time.time()
        tr, te = next(GroupShuffleSplit(1, test_size=ts, random_state=100 + r).split(table, y, groups=table["id_student"]))
        sp = dp.fit_encoder(feats.loc[tr], use_vle)
        X = dp.encode(feats, sp)
        rfe = RFE(RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5, n_jobs=-1,
                                         random_state=cfg.RANDOM_STATE), n_features_to_select=n_opt, step=2).fit(X.loc[tr], y.loc[tr])
        cols = [c for c, k in zip(X.columns, rfe.support_) if k]
        cat = [i for i, c in enumerate(cols) if c in sp["binary"] or "=" in c]
        Xf, yf = X.loc[tr, cols], y.loc[tr]
        if meta["imbalance"] == "smotenc":
            Xf, yf = (SMOTENC(categorical_features=cat, random_state=cfg.RANDOM_STATE) if cat
                      else SMOTE(random_state=cfg.RANDOM_STATE)).fit_resample(Xf, yf)
        m = make().fit(Xf, yf)
        p = m.predict_proba(X.loc[te, cols])[:, 1]
        acc.append(accuracy_score(y.loc[te], p >= .5))
        auc.append(roc_auc_score(y.loc[te], p))
        print(f"{label} repeat {r + 1}/{a.repeats}: acc={acc[-1]:.3f} auc={auc[-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    rows.append(dict(split=label, repeats=a.repeats, acc_mean=float(np.mean(acc)), acc_sd=float(np.std(acc, ddof=1)),
                     auc_mean=float(np.mean(auc)), auc_sd=float(np.std(auc, ddof=1))))
out = dict(model=name, rows=rows)
(cfg.ARTIFACT_DIR / "split_noise.json").write_text(json.dumps(out, indent=2))
print(pd.DataFrame(rows).round(4).to_string(index=False))
