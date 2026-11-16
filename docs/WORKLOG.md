# Work Log

A dated record of what was done, what was decided and why. Newest entries are at the bottom.

---

## 2026-11-16 — M0: Plan and scaffold

**Done**
- Created the repository structure and wrote [`PLAN.md`](PLAN.md). The hypotheses,
  primary metric (macro-F1), α = 0.05, Holm correction, MPD Δ = 0.01 and decision rule are
  all fixed *before* any model is trained.
- Environment: Python 3.13, scikit-learn 1.9, LightGBM 4.7, SciPy 1.18, statsmodels 0.15, mlxtend 0.25.

**Decisions**
- **Dataset: Dry Bean (UCI 602).** Multi-class (7), moderately imbalanced, not saturated.
  Iris/Wine/Digits were rejected because every model scores ≈ 97%+, so an A/B test has nothing to find.
- **Primary test: corrected resampled t-test on 10×10 CV.** A naive CV t-test is
  anti-conservative because training folds overlap (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004).
- **The data is not committed.** It is downloaded and checksum-verified instead (CC BY 4.0;
  keeps the repo small and the provenance explicit).

**First data check (profiling the raw file)**
- 13,611 rows, 16 features, 7 classes, 0 missing values.
- **68 exact duplicate rows**. These will be dropped (see PLAN §2).
- Class counts: DERMASON 3546, SIRA 2636, SEKER 2027, HOROZ 1928, CALI 1630, BARBUNYA 1322, BOMBAY 522.
