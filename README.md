# Student Dropout Early-Warning System (OULAD)

Decision Tree / Random Forest / XGBoost + RFE + SHAP, shown in a Streamlit dashboard.

## Preprocessing (follows the proposal)
- Files merged: studentInfo, studentRegistration, studentAssessment, assessments, courses, vle (+ **studentVle**, optional).
- Target: Withdrawn / Fail = at-risk (1), Pass / Distinction = not at-risk (0). `final_result` and `date_unregistration` are never features.
- Early-warning point = exam day (or presentation length) - 5 weeks. Only information available up to then is used.
  Students already unregistered before that point are excluded (nothing left to predict).
- **GroupShuffleSplit 80/20 by `id_student`** (one student never appears in both train and test; asserted in code).
- Encoding: one-hot (module, presentation, region), ordinal (age band, education, IMD), binary (gender, disability).
- Grouped cross-validation (StratifiedGroupKFold), **SMOTENC on training folds only**, RFE (10-15 features), grid search.
- Risk tiers: Low < 40%, Medium 40-69%, High >= 70%.
- Gender, disability and region are excluded from the model by default (`EXCLUDE_SENSITIVE` in `core/config.py`).

## studentVle.csv (too big for GitHub)
Put the file in `data/studentVle.csv` on **your computer only** (it is in `.gitignore`), then retrain:

    python train.py

It is read in chunks (low RAM) and turned into: total clicks, active days, distinct sites, clicks per active day,
recent-4-week share, and clicks per activity type. If the file is missing, training continues without click features.
The shipped `artifacts/` were trained **without** it.

## Run
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt
    python train.py          # --fast for a quick test
    python tests/test_consistency.py
    streamlit run app.py

## Deploy (Render)
Commit `artifacts/` (not `data/`), push to GitHub, create a Web Service from the repo (`render.yaml` is included).
Retrain locally with the same library versions as `requirements.txt` before committing artifacts.
The app never reads the raw OULAD files at runtime.
