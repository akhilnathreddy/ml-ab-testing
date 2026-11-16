"""Experiment configuration.

Everything that defines the experiment lives here, in one place, so the design in
``docs/PLAN.md`` maps one-to-one onto code. These values were fixed before the
experiment was run (pre-registration). Do not tune them after looking at results.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --------------------------------------------------------------------------- data
DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DATASET_ZIP = DATA_DIR / "dry_bean_dataset.zip"
DATASET_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
DATASET_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
TARGET = "Class"

# --------------------------------------------------------------------------- design
SEED = 42
HOLDOUT_SIZE = 0.20  # stratified hold-out, touched once at the very end

CV_REPEATS = 10  # 10 x 10 repeated stratified K-fold -> 100 paired scores
CV_FOLDS = 10
FIVE_BY_TWO_REPEATS = 5  # Dietterich's 5x2cv

PRIMARY_METRIC = "f1_macro"
ALPHA = 0.05
MPD = 0.01  # minimum practically meaningful difference in macro-F1 (1 pp)
N_BOOTSTRAP = 10_000

# Arm labels used throughout tables and figures (A = control).
MODEL_ORDER = ["LogReg", "RandomForest", "LightGBM"]
COMPARISONS = [
    ("RandomForest", "LogReg"),  # B vs A
    ("LightGBM", "LogReg"),  # C vs A
    ("LightGBM", "RandomForest"),  # C vs B
]

N_JOBS = -1  # parallelism across CV folds (each model itself is single-threaded)
