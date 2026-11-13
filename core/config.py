"""Central settings. Change values here, then re-run:  python train.py"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
VLE_FILE = Path(os.environ.get("OULAD_VLE_FILE", DATA_DIR / "studentVle.csv"))   # too big for GitHub: keep it on your computer only

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---- Early prediction point (proposal 3.4.2: 4-5 weeks before the final exam) -------
# cutoff day = exam day (or end of presentation when the exam date is blank) - WEEKS_BEFORE_EXAM * 7
# Only VLE clicks, assessments and registration known up to this day are used.
WEEKS_BEFORE_EXAM = 5

# Students who had already unregistered on/before the cutoff are already known withdrawals, so they are
# not "early warning" cases. date_unregistration is used ONLY for this eligibility rule, never as a feature.
EXCLUDE_ALREADY_WITHDRAWN = True

# Identity attributes are left out of the model so explanations never blame who the student is.
EXCLUDE_SENSITIVE = True
SENSITIVE_COLUMNS = ["gender", "disability", "region"]

# "smotenc"      -> SMOTENC on the TRAIN folds only (proposal 3.5.2). Test data is never touched.
# "class_weight" -> class_weight='balanced' / scale_pos_weight
# "none"         -> no balancing
IMBALANCE = "smotenc"

# ---- Risk tiers (proposal 3.5.3) ----------------------------------------------------
LOW_MAX = 0.40
HIGH_MIN = 0.70

# ---- Pre-processing / feature selection ----------------------------------------------
RFE_MIN_FEATURES = 10
RFE_MAX_FEATURES = 15
RFE_TOLERANCE = 0.003
RFE_STEP = 2
SHAP_BACKGROUND_SIZE = 100
VLE_CHUNK_ROWS = 2_000_000
VLE_TYPES = ["homepage", "forumng", "oucontent", "subpage", "resource", "url", "quiz"]  # rest -> "other"
RECENT_DAYS = 28                    # "recent" = the last 4 weeks before the cutoff

# ---- Notifications ------------------------------------------------------------------
DEFAULT_COUNTRY_CODE = "60"
