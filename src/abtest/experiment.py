"""Run the experiment: repeated CV, 5x2cv, hold-out evaluation and all tests."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from . import config, stats_tests
from .data import Dataset, load_dataset
from .metrics import METRICS, compute_metrics, fast_macro_f1
from .models import MODEL_DESCRIPTIONS, make_models


# --------------------------------------------------------------------------- CV engine
def _fit_score(name, model, X, y, train_idx, test_idx, labels, meta) -> dict:
    t0 = time.perf_counter()
    est = clone(model).fit(X.iloc[train_idx], y[train_idx])
    fit_s = time.perf_counter() - t0
    Xt, yt = X.iloc[test_idx], y[test_idx]
    pred = est.predict(Xt)
    proba = est.predict_proba(Xt)
    row = {
        **meta,
        "model": name,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "fit_seconds": fit_s,
    }
    row.update(compute_metrics(yt, pred, proba, labels))
    return row


def _run_splits(models, X, y, splits, labels, n_jobs) -> pd.DataFrame:
    jobs = [
        delayed(_fit_score)(name, model, X, y, tr, te, labels, meta)
        for meta, (tr, te) in splits
        for name, model in models.items()
    ]
    rows = Parallel(n_jobs=n_jobs)(jobs)
    return pd.DataFrame(rows)


def repeated_cv(ds: Dataset, models, n_jobs=config.N_JOBS) -> pd.DataFrame:
    """10x10 repeated stratified K-fold. Every model sees *identical* folds."""
    labels = np.arange(len(ds.classes))
    rskf = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=config.SEED
    )
    splits = [
        ({"repeat": i // config.CV_FOLDS, "fold": i % config.CV_FOLDS}, s)
        for i, s in enumerate(rskf.split(ds.X_dev, ds.y_dev))
    ]
    return _run_splits(models, ds.X_dev, ds.y_dev, splits, labels, n_jobs)


def five_by_two_cv(ds: Dataset, models, n_jobs=config.N_JOBS) -> pd.DataFrame:
    """Dietterich's 5x2cv: 5 repetitions of stratified 2-fold CV."""
    labels = np.arange(len(ds.classes))
    splits = []
    for i in range(config.FIVE_BY_TWO_REPEATS):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.SEED + i)
        for j, s in enumerate(skf.split(ds.X_dev, ds.y_dev)):
            splits.append(({"repeat": i, "fold": j}, s))
    return _run_splits(models, ds.X_dev, ds.y_dev, splits, labels, n_jobs)


def _paired_diffs(df: pd.DataFrame, m1: str, m2: str, metric: str) -> pd.Series:
    wide = df.pivot_table(index=["repeat", "fold"], columns="model", values=metric)
    return (wide[m1] - wide[m2]).sort_index()


# --------------------------------------------------------------------------- results
@dataclass
class Results:
    dataset: Dataset
    cv: pd.DataFrame
    cv_5x2: pd.DataFrame
    holdout_metrics: pd.DataFrame
    holdout_preds: dict[str, np.ndarray]
    holdout_proba: dict[str, np.ndarray]
    cv_tests: pd.DataFrame
    holdout_tests: pd.DataFrame
    secondary_tests: pd.DataFrame
    bootstrap_dists: dict[str, np.ndarray]
    decisions: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _decide(row: pd.Series) -> str:
    sig = row["p_holm"] < config.ALPHA
    big = abs(row["mean_diff"]) >= config.MPD
    winner = row["model_1"] if row["mean_diff"] > 0 else row["model_2"]
    if sig and big:
        return f"Prefer {winner} (significant and ≥ MPD)"
    if sig and row["tost_equivalent"]:
        return (f"{winner} is significantly better, but by < MPD and TOST-equivalent "
                "→ practically equivalent")
    if row["tost_equivalent"]:
        return "Not significant and TOST-equivalent → practically equivalent"
    if sig:
        return f"{winner} significantly better, but by < MPD → not practically meaningful"
    return "No significant difference, equivalence not shown → inconclusive"


def run(n_jobs: int = config.N_JOBS, n_boot: int = config.N_BOOTSTRAP) -> Results:
    t_start = time.perf_counter()
    ds = load_dataset()
    models = make_models(config.SEED)
    labels = np.arange(len(ds.classes))

    print("Running 10x10 repeated stratified CV ...")
    cv = repeated_cv(ds, models, n_jobs)
    print("Running 5x2cv ...")
    cv52 = five_by_two_cv(ds, models, n_jobs)

    print("Fitting on the dev set and scoring the hold-out set ...")
    ho_rows, preds, probas = [], {}, {}
    for name, model in models.items():
        est = clone(model).fit(ds.X_dev, ds.y_dev)
        preds[name] = est.predict(ds.X_holdout)
        probas[name] = est.predict_proba(ds.X_holdout)
        ho_rows.append(
            {"model": name, **compute_metrics(ds.y_holdout, preds[name], probas[name], labels)}
        )
    holdout = pd.DataFrame(ho_rows).set_index("model").loc[config.MODEL_ORDER]

    n_train = cv["n_train"].mean()
    n_test = cv["n_test"].mean()
    pm = config.PRIMARY_METRIC

    # ---- primary family: CV tests on macro-F1 -------------------------------------
    cv_rows = []
    for m1, m2 in config.COMPARISONS:
        d = _paired_diffs(cv, m1, m2, pm).to_numpy()
        d52 = _paired_diffs(cv52, m1, m2, pm).to_numpy().reshape(5, 2)
        corr = stats_tests.corrected_resampled_ttest(d, n_train, n_test, config.ALPHA)
        naive = stats_tests.naive_paired_ttest(d, config.ALPHA)
        tost = stats_tests.tost_corrected(d, n_train, n_test, config.MPD, config.ALPHA)
        f52 = stats_tests.five_by_two_cv_ttest(d52)
        cv_rows.append(
            {
                "comparison": f"{m1} vs {m2}",
                "model_1": m1,
                "model_2": m2,
                "mean_diff": corr["mean_diff"],
                "ci_low": corr["ci_low"],
                "ci_high": corr["ci_high"],
                "cohens_dz": stats_tests.cohens_dz(d),
                "wins_1": int((d > 0).sum()),
                "wins_2": int((d < 0).sum()),
                "t_corrected": corr["t"],
                "p_corrected": corr["p_value"],
                "naive_ci_low": naive["ci_low"],
                "naive_ci_high": naive["ci_high"],
                "t_naive": naive["t"],
                "p_naive": naive["p_value"],
                "t_5x2": f52["t"],
                "p_5x2": f52["p_value"],
                "f_5x2_combined": f52["f_combined"],
                "p_5x2_combined_f": f52["p_value_f_combined"],
                "tost_p": tost["p_value"],
                "tost_equivalent": tost["equivalent"],
                "tost_ci90_low": tost["ci90_low"],
                "tost_ci90_high": tost["ci90_high"],
            }
        )
    cv_tests = pd.DataFrame(cv_rows)
    for col in ["p_corrected", "p_naive", "p_5x2", "p_5x2_combined_f"]:
        _, cv_tests[col.replace("p_", "p_holm_", 1)] = stats_tests.holm(cv_tests[col], config.ALPHA)
    cv_tests["p_holm"] = cv_tests["p_holm_corrected"]
    cv_tests["decision"] = cv_tests.apply(_decide, axis=1)
    cv_tests = cv_tests.drop(columns="p_holm")

    # ---- hold-out family: McNemar + bootstrap ----------------------------------------
    k = len(ds.classes)
    ho_rows, boot = [], {}
    for m1, m2 in config.COMPARISONS:
        mc = stats_tests.mcnemar_test(ds.y_holdout, preds[m1], preds[m2])
        bs = stats_tests.paired_bootstrap(
            ds.y_holdout,
            preds[m1],
            preds[m2],
            lambda yt, yp: fast_macro_f1(yt, yp, k),
            n_boot=n_boot,
            alpha=config.ALPHA,
            seed=config.SEED,
        )
        boot[f"{m1} vs {m2}"] = bs.pop("distribution")
        ho_rows.append(
            {
                "comparison": f"{m1} vs {m2}",
                "model_1": m1,
                "model_2": m2,
                "both_correct": mc["both_correct"],
                "only_1_correct": mc["only_1_correct"],
                "only_2_correct": mc["only_2_correct"],
                "both_wrong": mc["both_wrong"],
                "p_mcnemar": mc["p_value"],
                "macro_f1_diff": bs["mean_diff"],
                "boot_ci_low": bs["ci_low"],
                "boot_ci_high": bs["ci_high"],
                "p_bootstrap": bs["p_value"],
            }
        )
    holdout_tests = pd.DataFrame(ho_rows)
    _, holdout_tests["p_holm_mcnemar"] = stats_tests.holm(holdout_tests["p_mcnemar"])
    _, holdout_tests["p_holm_bootstrap"] = stats_tests.holm(holdout_tests["p_bootstrap"])

    # ---- secondary metrics (exploratory): corrected t-test on every metric ----------
    sec_rows = []
    for metric, (label, higher) in METRICS.items():
        for m1, m2 in config.COMPARISONS:
            d = _paired_diffs(cv, m1, m2, metric).to_numpy()
            r = stats_tests.corrected_resampled_ttest(d, n_train, n_test, config.ALPHA)
            sec_rows.append(
                {
                    "metric": metric,
                    "metric_label": label,
                    "higher_is_better": higher,
                    "comparison": f"{m1} vs {m2}",
                    "mean_diff": r["mean_diff"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "p_corrected": r["p_value"],
                }
            )
    secondary = pd.DataFrame(sec_rows)
    secondary["p_holm_within_metric"] = secondary.groupby("metric")["p_corrected"].transform(
        lambda p: stats_tests.holm(p)[1]
    )

    decisions = cv_tests[["comparison", "mean_diff", "ci_low", "ci_high", "p_holm_corrected",
                          "tost_equivalent", "decision"]]

    meta = {
        "seed": config.SEED,
        "n_raw_rows": ds.n_raw,
        "n_duplicates_dropped": ds.n_duplicates,
        "n_dev": int(len(ds.y_dev)),
        "n_holdout": int(len(ds.y_holdout)),
        "classes": ds.classes,
        "cv": f"{config.CV_REPEATS}x{config.CV_FOLDS} repeated stratified K-fold",
        "mean_n_train": float(n_train),
        "mean_n_test": float(n_test),
        "primary_metric": pm,
        "alpha": config.ALPHA,
        "mpd": config.MPD,
        "n_bootstrap": n_boot,
        "models": MODEL_DESCRIPTIONS,
        "python": platform.python_version(),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    return Results(ds, cv, cv52, holdout, preds, probas, cv_tests, holdout_tests,
                   secondary, boot, decisions, meta)


def save(res: Results) -> None:
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    res.cv.to_csv(out / "cv_fold_scores.csv", index=False)
    res.cv_5x2.to_csv(out / "cv5x2_fold_scores.csv", index=False)

    summary = res.cv.groupby("model")[list(METRICS)].agg(["mean", "std"]).loc[config.MODEL_ORDER]
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.to_csv(out / "cv_metrics_summary.csv")

    res.holdout_metrics.to_csv(out / "holdout_metrics.csv")
    res.cv_tests.to_csv(out / "stat_tests_cv.csv", index=False)
    res.holdout_tests.to_csv(out / "stat_tests_holdout.csv", index=False)
    res.secondary_tests.to_csv(out / "stat_tests_secondary_metrics.csv", index=False)
    res.decisions.to_csv(out / "decisions.csv", index=False)

    preds = pd.DataFrame({"y_true": res.dataset.y_holdout, **res.holdout_preds})
    preds.to_csv(out / "holdout_predictions.csv", index=False)

    import sklearn, lightgbm, scipy, statsmodels  # noqa: E401 (versions only)

    res.meta["versions"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }
    (out / "run_metadata.json").write_text(json.dumps(res.meta, indent=2, ensure_ascii=False))
