"""Unit tests for the statistical machinery.

Hand-rolled tests are checked against reference implementations (mlxtend,
statsmodels, sklearn) or closed-form values wherever possible.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from abtest import stats_tests as st
from abtest.metrics import compute_metrics, fast_macro_f1

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- naive / corrected t
def test_naive_matches_scipy_ttest_rel():
    a, b = RNG.normal(0.80, 0.02, 30), RNG.normal(0.79, 0.02, 30)
    ours = st.naive_paired_ttest(a - b)
    ref = stats.ttest_rel(a, b)
    assert ours["t"] == pytest.approx(ref.statistic)
    assert ours["p_value"] == pytest.approx(ref.pvalue)


def test_corrected_ttest_formula():
    d = RNG.normal(0.01, 0.01, 100)
    n_train, n_test = 900, 100
    r = st.corrected_resampled_ttest(d, n_train, n_test)
    se = np.sqrt((1 / 100 + n_test / n_train) * d.var(ddof=1))
    assert r["t"] == pytest.approx(d.mean() / se)
    assert r["p_value"] == pytest.approx(2 * stats.t.sf(abs(d.mean() / se), 99))
    assert r["ci_low"] < d.mean() < r["ci_high"]


def test_corrected_is_more_conservative_than_naive():
    d = RNG.normal(0.005, 0.01, 100)
    corr = st.corrected_resampled_ttest(d, 9000, 1000)
    naive = st.naive_paired_ttest(d)
    assert corr["p_value"] > naive["p_value"]
    assert (corr["ci_high"] - corr["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_corrected_ttest_zero_variance():
    r = st.corrected_resampled_ttest(np.zeros(10), 90, 10)
    assert r["p_value"] == 1.0


# ---------------------------------------------------------------- TOST
def test_tost_equivalent_when_ci_inside_margin():
    d = RNG.normal(0.0, 0.002, 100)
    r = st.tost_corrected(d, 9000, 1000, margin=0.01)
    assert r["equivalent"]
    assert -0.01 < r["ci90_low"] < r["ci90_high"] < 0.01


def test_tost_not_equivalent_for_large_effect():
    d = RNG.normal(0.03, 0.005, 100)
    assert not st.tost_corrected(d, 9000, 1000, margin=0.01)["equivalent"]


# ---------------------------------------------------------------- 5x2cv vs mlxtend
def _mlxtend_style_diffs(est1, est2, X, y, seed):
    """Replicates mlxtend's split scheme so the statistic can be compared exactly."""
    rng = np.random.RandomState(seed)
    diffs = np.empty((5, 2))
    for i in range(5):
        r = rng.randint(low=0, high=32767)
        X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=r)
        for j, (Xa, Xb, ya, yb) in enumerate([(X1, X2, y1, y2), (X2, X1, y2, y1)]):
            s1 = est1.fit(Xa, ya).score(Xb, yb)
            s2 = est2.fit(Xa, ya).score(Xb, yb)
            diffs[i, j] = s1 - s2
    return diffs


def test_5x2cv_matches_mlxtend():
    from mlxtend.evaluate import combined_ftest_5x2cv, paired_ttest_5x2cv

    X, y = load_iris(return_X_y=True)
    e1 = LogisticRegression(max_iter=1000)
    e2 = DecisionTreeClassifier(max_depth=1, random_state=0)
    t_ref, p_ref = paired_ttest_5x2cv(e1, e2, X, y, random_seed=1)
    f_ref, pf_ref = combined_ftest_5x2cv(e1, e2, X, y, random_seed=1)

    ours = st.five_by_two_cv_ttest(_mlxtend_style_diffs(e1, e2, X, y, seed=1))
    assert ours["t"] == pytest.approx(t_ref)
    assert ours["p_value"] == pytest.approx(p_ref)
    assert ours["f_combined"] == pytest.approx(f_ref)
    assert ours["p_value_f_combined"] == pytest.approx(pf_ref)


def test_5x2cv_shape_check():
    with pytest.raises(ValueError):
        st.five_by_two_cv_ttest(np.zeros((10,)))


# ---------------------------------------------------------------- McNemar
def test_mcnemar_counts_and_exact_p():
    y = np.array([0] * 100)
    p1 = np.array([0] * 90 + [1] * 10)  # 90 correct
    p2 = np.array([0] * 80 + [1] * 10 + [0] * 10)  # 90 correct, errors in other places
    r = st.mcnemar_test(y, p1, p2)
    assert (r["both_correct"], r["only_1_correct"], r["only_2_correct"], r["both_wrong"]) == (
        80, 10, 10, 0,
    )
    # exact McNemar == two-sided binomial test on the discordant pairs
    assert r["p_value"] == pytest.approx(stats.binomtest(10, 20, 0.5).pvalue)


def test_mcnemar_detects_real_difference():
    y = np.zeros(1000, dtype=int)
    p1 = np.zeros(1000, dtype=int)
    p2 = np.where(np.arange(1000) < 60, 1, 0)
    assert st.mcnemar_test(y, p1, p2)["p_value"] < 1e-10


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_identical_models_is_null():
    y = RNG.integers(0, 3, 500)
    p = np.where(RNG.random(500) < 0.8, y, (y + 1) % 3)
    r = st.paired_bootstrap(y, p, p, lambda a, b: fast_macro_f1(a, b, 3), n_boot=500)
    assert r["mean_diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["p_value"] == 1.0


def test_bootstrap_ci_covers_observed():
    y = RNG.integers(0, 3, 800)
    p1 = np.where(RNG.random(800) < 0.90, y, (y + 1) % 3)
    p2 = np.where(RNG.random(800) < 0.80, y, (y + 1) % 3)
    r = st.paired_bootstrap(y, p1, p2, lambda a, b: fast_macro_f1(a, b, 3), n_boot=2000)
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    assert r["ci_low"] > 0 and r["p_value"] < 0.01


# ---------------------------------------------------------------- Holm
def test_holm_matches_manual():
    p = [0.01, 0.04, 0.03]
    reject, adj = st.holm(p)
    # sorted: 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    assert list(reject) == [True, False, False]


# ---------------------------------------------------------------- metrics
def test_fast_macro_f1_matches_sklearn():
    y = RNG.integers(0, 7, 3000)
    p = np.where(RNG.random(3000) < 0.7, y, RNG.integers(0, 7, 3000))
    assert fast_macro_f1(y, p, 7) == pytest.approx(f1_score(y, p, average="macro"))


def test_hamming_loss_equals_one_minus_accuracy_for_single_label():
    y = RNG.integers(0, 7, 1000)
    p = np.where(RNG.random(1000) < 0.85, y, RNG.integers(0, 7, 1000))
    m = compute_metrics(y, p, labels=np.arange(7))
    assert m["hamming_loss"] == pytest.approx(1 - m["accuracy"])


def test_cohens_dz():
    d = np.array([1.0, 2.0, 3.0])
    assert st.cohens_dz(d) == pytest.approx(2.0 / 1.0)
