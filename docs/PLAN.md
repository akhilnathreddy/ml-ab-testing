# Project Plan — ML A/B Testing on a Multi-class Benchmark

> **Brief (Data Analyst – ML Tester):** Run one end-to-end A/B test (t-tests) between 2–3
> models on a benchmark dataset, with multiple metrics (F1, Precision, Recall, Hamming
> Loss, …), and compare them with statistical tests. Deliverable: a report/blog URL.

This document is the design. It was written **before** any model was trained, so the
hypotheses, primary metric, α and decision rule are fixed in advance (pre-registration).
Progress against the plan is tracked in [`WORKLOG.md`](WORKLOG.md).

---

## 1. Research question

> On the Dry Bean benchmark, does a more complex model (Random Forest or LightGBM)
> classify beans **meaningfully** better than a simple Logistic Regression baseline, and
> does boosting beat bagging?

"Meaningfully" means **both** statistically significant **and** above a minimum
practical effect size (defined in §5).

## 2. Dataset

| Item | Value |
|---|---|
| Name | Dry Bean Dataset (Koklu & Ozkan, 2020) |
| Source | UCI ML Repository, id 602 — <https://archive.ics.uci.edu/dataset/602/dry+bean+dataset> |
| License | CC BY 4.0 |
| Size | 13,611 rows × 16 numeric shape features |
| Target | 7 bean varieties (multi-class, imbalanced: 522 → 3,546 per class) |
| Why this one | Not saturated: models genuinely differ, unlike Iris/Wine/Digits where everything scores ≈ 97–100%. |

Data handling:
- Downloaded by script, verified by SHA-256 checksum, **not committed** to git (`data/` is ignored).
- 68 exact duplicate rows are dropped so identical rows can't land in both a train and a test fold (leakage).
- One stratified **80/20 split** (seed 42): the 80% *development* set is used for all
  cross-validation; the 20% *hold-out* set is touched once, at the end.

## 3. Models (the "variants")

| Arm | Model | Role |
|---|---|---|
| A | Logistic Regression (StandardScaler → multinomial LR) | Control / baseline |
| B | Random Forest (500 trees) | Challenger — bagging |
| C | LightGBM (gradient-boosted trees) | Challenger — boosting |

- Every preprocessing step sits inside an sklearn `Pipeline`, so scaling is fitted on
  training folds only.
- Hyperparameters are **fixed in advance** with sensible values (no tuning). Tuning would
  require nested CV to stay unbiased; this is listed as a limitation.
- All randomness is seeded.

## 4. Metrics

Computed on every test fold and on the hold-out set:

| Metric | Notes |
|---|---|
| **Macro F1** | **Primary metric.** Treats all 7 classes equally; matters because of imbalance. |
| Accuracy | |
| Macro / Weighted Precision | |
| Macro / Weighted Recall | |
| Weighted F1 | |
| Hamming Loss | For single-label multi-class it equals `1 − accuracy`. It is reported (as the brief asks), but the report explains why it adds no new information here. |
| MCC (Matthews corr.) | Robust single-number summary under imbalance |
| Cohen's κ | Agreement beyond chance |
| Log loss | Quality of predicted probabilities |

## 5. Hypotheses & decision rule (pre-registered)

Three pairwise comparisons: **B vs A**, **C vs A**, **C vs B**.

- **H₀:** mean macro-F1 of the two models is equal. **H₁:** they differ (two-sided).
- **α = 0.05**, family-wise error controlled with **Holm–Bonferroni** across the 3 pairs.
- **Minimum practically meaningful difference (MPD): Δ = 0.01** (1 percentage point of macro-F1).
- **Decision rule** — prefer the challenger only if **both** of these hold:
  1. The Holm-adjusted *p* < 0.05 on the primary test (corrected resampled t-test).
  2. The observed mean improvement is ≥ Δ.
- **Equivalence check (TOST, ±Δ):** if the models are "practically equivalent", prefer
  the simpler or cheaper one.

## 6. Statistical tests

| # | Test | Data | Why |
|---|---|---|---|
| 1 | **Corrected resampled t-test** (Nadeau & Bengio, 2003) — **primary** | 10×10 repeated stratified CV on the dev set (100 paired fold scores) | A naive paired t-test on CV folds inflates Type-I error, because training sets overlap. The `(1/k + n_test/n_train)` variance correction fixes this. |
| 2 | **5×2cv paired t-test** (Dietterich, 1998) | 5 repetitions × 2-fold CV on the dev set | Independent second opinion with a well-studied Type-I error rate. |
| 3 | **McNemar's test** (exact) | Hold-out predictions | Tests whether the two models make *different errors on the same samples*. |
| 4 | **Paired bootstrap** (10,000 resamples) | Hold-out predictions | 95% CI for the macro-F1 difference on unseen data. |
| 5 | **TOST equivalence** | Same CV data as test 1 | Can we claim the models are practically the same? |
| — | **Naive paired t-test** (for illustration only) | Same CV data as test 1 | Shown side by side to demonstrate *why* it's wrong (it gives far smaller p-values). |

Effect sizes: mean difference, 95% CI (with corrected variance), and Cohen's d_z.

## 7. Deliverables

| Deliverable | Location |
|---|---|
| Report (GitHub) | [`README.md`](../README.md) |
| Blog post draft (Medium-ready) | [`docs/BLOG_POST.md`](BLOG_POST.md) |
| Methodology deep dive | [`docs/METHODOLOGY.md`](METHODOLOGY.md) |
| Work log / decisions | [`docs/WORKLOG.md`](WORKLOG.md) |
| Reproducible pipeline | `src/abtest/`, `scripts/run_experiment.py`, `Makefile` |
| Executed notebook | `notebooks/analysis.ipynb` |
| Results tables & figures | `results/`, `results/figures/` |
| Unit tests for the stats code | `tests/` |

## 8. Milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Plan, repo scaffold, documentation skeleton | ✅ |
| M1 | Data download + validation + EDA | ✅ |
| M2 | Models and metrics, CV engine | ✅ |
| M3 | Statistical tests + unit tests (incl. cross-check vs `mlxtend`) | ✅ |
| M4 | Full experiment run, results tables, figures | ✅ |
| M5 | Report (README), methodology, blog draft, executed notebook | ✅ |
| M6 | Review, merge to `main`, push | ✅ |

## 9. Known limitations (to discuss in the report)

- One dataset only. Conclusions are about Dry Bean, not models in general (a Friedman/Nemenyi test needs many datasets).
- Fixed hyperparameters. A tuned LR or RF might close the gap.
- The corrected t-test is an approximation. Its correction factor is a heuristic that is conservative in practice.
- The hold-out set is a single split, so McNemar and the bootstrap reflect one sample of test data.
