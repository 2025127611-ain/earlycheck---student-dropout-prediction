"""Loads the trained artifacts and exposes ONE prediction path for Home, SHAP Explanation and Live Prediction.

The app never needs the raw OULAD files (studentVle.csv is far too big for GitHub / Render): everything the
dashboard shows for the test-set students is saved by train.py in artifacts/.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from . import data_prep as dp
from . import explain as ex


@dataclass
class Result:
    prob: float
    level: str
    base: float
    table: pd.DataFrame        # per-variable contributions (percentage points), sorted by size
    features: pd.Series


class Predictor:
    def __init__(self, art_dir: Path = cfg.ARTIFACT_DIR):
        art_dir = Path(art_dir)
        self.meta = json.loads((art_dir / "meta.json").read_text())
        self.model = joblib.load(art_dir / "model.joblib")
        self.spec = joblib.load(art_dir / "encoder.joblib")
        self.columns: list[str] = self.meta["selected_columns"]
        self.use_vle: bool = self.meta["use_vle"]
        self.units = dp.unit_map(self.use_vle)
        self.required: list[str] = self.meta["required_inputs"]
        self.students = pd.read_csv(art_dir / "students.csv")                       # all eligible enrolments
        self.test_inputs = pd.read_csv(art_dir / "test_inputs.csv", index_col="student_id")
        self.test_feats = pd.read_csv(art_dir / "test_feats.csv", index_col="student_id")
        self.test_shap = np.load(art_dir / "test_shap.npy")
        self.background = pd.read_csv(art_dir / "background.csv")
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = ex.build_explainer(self.model, self.background)
        return self._explainer

    @property
    def base(self) -> float:
        return float(self.meta["base_value"])

    def to_matrix(self, inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        feats = dp.engineer(inputs)
        X = dp.encode(feats, self.spec, only=self.columns)
        return feats, X

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        _, X = self.to_matrix(inputs)
        return self.model.predict_proba(X)[:, 1]

    def _closed(self, agg: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
        """Remove the tiny float residual so that base + sum(contributions) == predicted risk exactly."""
        resid = prob - (self.base + agg.sum(axis=1).values)
        w = agg.abs().values
        w = w / np.where(w.sum(axis=1, keepdims=True) == 0, 1, w.sum(axis=1, keepdims=True))
        return pd.DataFrame(agg.values + w * resid[:, None], columns=agg.columns, index=agg.index)

    def explain_inputs(self, inputs: pd.DataFrame) -> list[Result]:
        """Live path: risk + SHAP for any INPUT rows."""
        feats, X = self.to_matrix(inputs)
        prob = self.model.predict_proba(X)[:, 1]
        sv = ex.shap_values(self.explainer, X)
        agg = self._closed(ex.aggregate_units(sv, self.columns), prob)
        out = []
        for i in range(len(X)):
            tbl = ex.local_table(agg.iloc[i], feats.iloc[i], self.units)
            out.append(Result(float(prob[i]), ex.risk_level(prob[i]), self.base, tbl, feats.iloc[i]))
        return out

    def explain_known(self, student_id: str) -> Result:
        """Stored explanation for a test-set student."""
        pos = self.test_feats.index.get_loc(student_id)
        prob = float(self.students.set_index("student_id").loc[student_id, "risk"])
        agg = self._closed(ex.aggregate_units(self.test_shap[[pos]], self.columns), np.array([prob]))
        feats = self.test_feats.iloc[pos]
        return Result(prob, ex.risk_level(prob), self.base, ex.local_table(agg.iloc[0], feats, self.units),
                      feats)
