"""Statistical tests for comparing two classifiers.

Every function takes *already computed* scores or predictions and returns a plain
dict, so the statistics stay separate from model training and are easy to unit-test.

Sign convention: a difference is always ``score(model_1) - score(model_2)``. For a
higher-is-better metric, a positive value means model_1 is better.

References
----------
- Nadeau, C. & Bengio, Y. (2003). Inference for the Generalization Error.
  Machine Learning 52, 239–281.
- Bouckaert, R. & Frank, E. (2004). Evaluating the Replicability of Significance
  Tests for Comparing Learning Algorithms. PAKDD.
- Dietterich, T. (1998). Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms. Neural Computation 10(7), 1895–1923.
- Alpaydin, E. (1999). Combined 5x2cv F Test for Comparing Supervised
  Classification Learning Algorithms. Neural Computation 11(8), 1885–1892.
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika 12, 153–157.
- Schuirmann, D. (1987). A comparison of the two one-sided tests procedure ...
  Journal of Pharmacokinetics and Biopharmaceutics 15, 657–680.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure.
  Scandinavian Journal of Statistics 6, 65–70.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------------------------------- helpers
def _t_summary(mean: float, se: float, df: int, alpha: float) -> dict:
    if se == 0:
        t_stat = np.inf * np.sign(mean) if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        half = 0.0
    else:
        t_stat = mean / se
        p = 2 * stats.t.sf(abs(t_stat), df)
        half = stats.t.ppf(1 - alpha / 2, df) * se
    return {
        "mean_diff": float(mean),
        "se": float(se),
        "t": float(t_stat),
        "df": int(df),
        "p_value": float(p),
        "ci_low": float(mean - half),
        "ci_high": float(mean + half),
    }


def cohens_dz(diffs: Sequence[float]) -> float:
    """Standardised mean of paired differences (descriptive effect size)."""
    d = np.asarray(diffs, dtype=float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") * np.sign(d.mean())


# --------------------------------------------------------------------------- CV t-tests
def naive_paired_ttest(diffs: Sequence[float], alpha: float = 0.05) -> dict:
    """Standard paired t-test on CV fold differences.

    **For illustration only.** It treats the J fold scores as independent, but the
    training sets overlap heavily, so the variance is underestimated and Type-I error
    is inflated (Dietterich 1998; Nadeau & Bengio 2003).
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    se = d.std(ddof=1) / np.sqrt(j)
    return {"test": "naive paired t-test", **_t_summary(d.mean(), se, j - 1, alpha)}


def corrected_resampled_ttest(
    diffs: Sequence[float], n_train: float, n_test: float, alpha: float = 0.05
) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    For J = r·k paired scores from r-times-repeated k-fold CV
    (Bouckaert & Frank 2004)::

        t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) ),   df = J - 1

    The extra ``n_test/n_train`` term inflates the variance to account for the
    correlation between scores that comes from overlapping training sets.
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    correction = 1.0 / j + n_test / n_train
    se = np.sqrt(correction * d.var(ddof=1))
    out = _t_summary(d.mean(), se, j - 1, alpha)
    return {"test": "corrected resampled t-test", "correction": float(correction), **out}


def tost_corrected(
    diffs: Sequence[float],
    n_train: float,
    n_test: float,
    margin: float,
    alpha: float = 0.05,
) -> dict:
    """TOST equivalence test using the corrected-resampled standard error.

    H0: |Δ| ≥ margin  vs  H1: |Δ| < margin. Equivalence is declared when both
    one-sided p-values are < α, which is the same as the (1 − 2α) CI lying inside
    ±margin.
    """
    d = np.asarray(diffs, dtype=float)
    j = d.size
    df = j - 1
    se = np.sqrt((1.0 / j + n_test / n_train) * d.var(ddof=1))
    mean = d.mean()
    p_lower = stats.t.sf((mean + margin) / se, df)  # H0: Δ ≤ -margin
    p_upper = stats.t.cdf((mean - margin) / se, df)  # H0: Δ ≥ +margin
    p = max(p_lower, p_upper)
    half90 = stats.t.ppf(1 - alpha, df) * se
    return {
        "test": "TOST (corrected SE)",
        "margin": float(margin),
        "p_value": float(p),
        "equivalent": bool(p < alpha),
        "ci90_low": float(mean - half90),
        "ci90_high": float(mean + half90),
    }


def five_by_two_cv_ttest(diffs_5x2: np.ndarray) -> dict:
    """Dietterich's 5x2cv paired t-test, plus Alpaydin's combined 5x2cv F-test.

    ``diffs_5x2[i, j]`` is the score difference on fold j (0/1) of repetition i.
    """
    p = np.asarray(diffs_5x2, dtype=float)
    if p.shape != (5, 2):
        raise ValueError(f"expected shape (5, 2), got {p.shape}")
    p_bar = p.mean(axis=1)
    s2 = ((p - p_bar[:, None]) ** 2).sum(axis=1)
    denom = np.sqrt(s2.mean())
    t_stat = p[0, 0] / denom if denom > 0 else np.inf
    p_t = 2 * stats.t.sf(abs(t_stat), 5)

    f_stat = (p**2).sum() / (2 * s2.sum()) if s2.sum() > 0 else np.inf
    p_f = stats.f.sf(f_stat, 10, 5)
    return {
        "test": "5x2cv paired t-test",
        "mean_diff": float(p.mean()),
        "t": float(t_stat),
        "df": 5,
        "p_value": float(p_t),
        "f_combined": float(f_stat),
        "p_value_f_combined": float(p_f),
    }


# --------------------------------------------------------------------------- hold-out
def mcnemar_test(y_true, pred_1, pred_2) -> dict:
    """Exact McNemar test on paired hold-out predictions.

    Builds the 2×2 table of (model_1 correct?, model_2 correct?). Only the
    discordant cells matter: b = model_1 right & model_2 wrong, c = the reverse.
    """
    y_true, pred_1, pred_2 = map(np.asarray, (y_true, pred_1, pred_2))
    ok1, ok2 = pred_1 == y_true, pred_2 == y_true
    both = int(np.sum(ok1 & ok2))
    only1 = int(np.sum(ok1 & ~ok2))
    only2 = int(np.sum(~ok1 & ok2))
    neither = int(np.sum(~ok1 & ~ok2))
    table = [[both, only1], [only2, neither]]
    exact = _sm_mcnemar(table, exact=True)
    chi2 = _sm_mcnemar(table, exact=False, correction=True)
    return {
        "test": "McNemar (exact)",
        "both_correct": both,
        "only_1_correct": only1,
        "only_2_correct": only2,
        "both_wrong": neither,
        "p_value": float(exact.pvalue),
        "chi2_cc": float(chi2.statistic),
        "p_value_chi2_cc": float(chi2.pvalue),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Paired percentile bootstrap of ``metric(pred_1) - metric(pred_2)``.

    The same resampled indices are used for both models, which keeps the pairing.
    The two-sided p-value is computed from the bootstrap distribution shifted
    to H0 (Δ = 0).
    """
    y_true, pred_1, pred_2 = map(np.asarray, (y_true, pred_1, pred_2))
    rng = np.random.default_rng(seed)
    n = y_true.size
    observed = metric(y_true, pred_1) - metric(y_true, pred_2)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        boots[b] = metric(yt, pred_1[idx]) - metric(yt, pred_2[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    shifted = boots - boots.mean()
    p = (np.sum(np.abs(shifted) >= abs(observed)) + 1) / (n_boot + 1)
    return {
        "test": "paired bootstrap",
        "mean_diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p),
        "n_boot": int(n_boot),
        "distribution": boots,
    }


# --------------------------------------------------------------------------- multiplicity
def holm(p_values: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm–Bonferroni step-down adjustment. Returns (reject, adjusted p)."""
    reject, p_adj, _, _ = multipletests(list(p_values), alpha=alpha, method="holm")
    return reject, p_adj
