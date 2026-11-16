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

---

## 2026-11-16 — M1–M3: Pipeline, metrics, statistical tests

**Done**
- `data.py`: download with SHA-256 verification, dedupe and a stratified 80/20 split (dev 10,834 / hold-out 2,709).
- `models.py`: the three arms with fixed hyperparameters. Each model is single-threaded and
  parallelised across folds, which keeps runs deterministic.
- `metrics.py`: 11 metrics, plus a vectorised macro-F1 for the bootstrap (about 90× faster than
  sklearn's `f1_score` on 2,709 samples (0.018 ms vs 1.6 ms per call), and checked equal to
  sklearn in the tests).
- `stats_tests.py`: corrected resampled t, naive t (demo only), 5×2cv t and combined F, exact
  McNemar, paired bootstrap, TOST and Holm.
- **16 unit tests pass.** The 5×2cv statistic and p-value match `mlxtend.evaluate.paired_ttest_5x2cv`
  and `combined_ftest_5x2cv` exactly when fed mlxtend's split scheme. The naive test matches
  `scipy.stats.ttest_rel`, and exact McNemar matches a two-sided binomial test.

**Findings / decisions**
- **All 68 duplicate rows are HOROZ beans** (1,928 → 1,860).
- Timing (single thread, one fold): LogReg 0.15 s, LightGBM 3.7 s, RandomForest 11.2 s. The
  full run takes about 7.7 min on 4 cores.
- The experiment's 5×2cv uses *stratified* 2-fold splits (`seed + i`), which is a small
  improvement over mlxtend's unstratified splits. The statistic itself is identical.

## 2026-11-16 — M4: Full experiment run

**Headline results (macro F1, 10×10 CV):** LogReg 0.9368 · RF 0.9369 · LightGBM 0.9411.

| Comparison | Δ | Holm p (corrected) | TOST ±0.01 | Decision |
|---|---|---|---|---|
| RF vs LogReg | +0.0001 | 0.952 | equivalent | practically equivalent |
| LightGBM vs LogReg | +0.0043 | 0.055 | equivalent | practically equivalent |
| LightGBM vs RF | +0.0042 | 0.046 | equivalent | significant but < MPD → practically equivalent |

- **Naive t-test** p (Holm) = 1.3e-11 / 4.0e-13 for the LightGBM comparisons. This illustrates
  the inflated Type-I error.
- **Hold-out:** the ranking flips (LogReg = RF 0.9302 > LightGBM 0.9286). Nothing is significant
  (McNemar p ≥ 0.28, every bootstrap CI includes 0).
- **Log loss** (exploratory) is the one large, significant difference: LogReg 0.214 < RF 0.275 < LightGBM 0.340.
- **Outcome under the pre-registered rule:** no challenger qualifies, so the recommendation is
  Logistic Regression (simplest, fastest, best calibrated).

**Change made after the first run (labelling only, no statistics changed):** the decision text
now distinguishes "significant **and** TOST-equivalent" from "not significant **and**
TOST-equivalent". The first version labelled both "practically equivalent", which hid the fact
that LightGBM vs RF is statistically significant.

**Reproducibility check:** a second full run from scratch (after the labelling change) produced
byte-identical CSVs for every fold score, metric and p-value.

## 2026-11-16 — M5: Figures, report, blog, notebook

- 10 figures in `results/figures/`, using a colour-blind-safe categorical palette that passes
  all-pairs CVD and normal-vision checks for 3 series. Every chart carries direct labels and
  the README repeats each number in a table, so no information relies on colour alone.
- Figure fixes after visual review: the all-metrics chart was rebuilt as small multiples (a single
  shared axis hid every difference), the per-class chart dodges overlapping points (BOMBAY = 1.0
  for every model), and tick and label collisions were fixed.
- `README.md` is the full report. `docs/BLOG_POST.md` is a Medium-ready version whose images
  are linked from GitHub raw URLs. `notebooks/analysis.ipynb` is executed against the cached results.
