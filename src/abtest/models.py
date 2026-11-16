"""The three arms of the A/B test.

Hyperparameters are fixed, sensible values chosen *before* seeing results (no
tuning). Each model is single-threaded; parallelism happens across CV folds,
which keeps runs deterministic for a given seed.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(seed: int) -> dict[str, BaseEstimator]:
    return {
        # A — control: linear baseline. Scaling lives inside the pipeline so it is
        # fitted on the training fold only (no leakage).
        "LogReg": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
            ]
        ),
        # B — bagging challenger.
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=1,
            random_state=seed,
        ),
        # C — boosting challenger.
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=seed,
            verbose=-1,
        ),
    }


MODEL_DESCRIPTIONS = {
    "LogReg": "StandardScaler → multinomial LogisticRegression(C=1, lbfgs)",
    "RandomForest": "RandomForest(500 trees, max_features='sqrt')",
    "LightGBM": "LightGBM(400 trees, lr=0.05, 31 leaves, subsample=0.8, colsample=0.8)",
}
