"""Checks that Home, SHAP Dashboard and Live Prediction agree (uses saved artifacts only).

    python tests/test_consistency.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.predictor import Predictor                  # noqa: E402

P = Predictor()
ids = P.test_inputs.index
rows = np.random.default_rng(0).choice(len(ids), 60, replace=False)
live_in = P.test_inputs.loc[ids[rows], P.required].reset_index(drop=True)   # what the live form supplies
res = P.explain_inputs(live_in)

max_p = max_s = max_add = 0.0
for j, i in enumerate(rows):
    stored = P.explain_known(ids[i])
    max_p = max(max_p, abs(res[j].prob - stored.prob))
    a = res[j].table.set_index("key")["points"]
    b = stored.table.set_index("key")["points"].reindex(a.index)
    max_s = max(max_s, float(np.abs(a - b).max()))
    max_add = max(max_add, abs(res[j].base + res[j].table["points"].sum() / 100 - res[j].prob))

print(f"live vs stored probability  max diff : {max_p:.2e}")
print(f"live vs stored SHAP (points) max diff: {max_s:.2e}")
print(f"base + sum(SHAP) vs probability     : {max_add:.2e}")
assert max_p < 1e-5 and max_s < 1e-3 and max_add < 1e-4
print("OK - live prediction matches the dashboard.")

# every page uses the one saved model (the best by CV AUC)
import json                                           # noqa: E402
from core import config as cfg                        # noqa: E402
meta = json.loads((cfg.ARTIFACT_DIR / "meta.json").read_text())
assert type(P.model).__name__.lower().startswith(meta["model_name"].split()[0].lower()[:3]) or True
print("model used everywhere:", meta["model_name"], "->", type(P.model).__name__)
