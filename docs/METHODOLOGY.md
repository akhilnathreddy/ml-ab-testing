# Methodology: How to A/B Test Machine-Learning Models Properly

This document explains *why* each piece of the experiment is set up the way it is.
The plan is in [`PLAN.md`](PLAN.md) and the results are in the [README](../README.md).

---

## 1. An ML A/B test vs a product A/B test

| Product A/B test | This ML A/B test |
|---|---|
| Users are randomly split into variants | **Data folds** are split, and every model sees *identical* folds |
| Unit = a user | Unit = a test fold (CV tests) or a test sample (McNemar/bootstrap) |
| Metric = conversion rate | Metric = **macro F1** (pre-registered) |
| Groups are independent | Scores are **paired**: same data, different model |

Pairing is the main advantage here. Every model is scored on exactly the same test fold,
so we analyse the **per-fold difference**. That cancels out how hard the fold happens to
be, which is usually the largest source of variance (see `results/figures/02_*.png`:
fold-to-fold spread ≫ between-model gaps).

## 2. Data protocol

```
13,611 raw rows ──drop 68 exact duplicates──► 13,543 rows
       │
       └─ stratified 80/20 split (seed 42)
              ├── Development set: 10,834 rows ─► 10×10 repeated stratified CV  (tests 1, 5, naive)
              │                                └► 5×2 stratified CV             (test 2)
              └── Hold-out set:     2,709 rows ─► touched once: McNemar + bootstrap (tests 3, 4)
```

- **Duplicates are dropped first.** Otherwise the same bean could appear in both a training
  fold and a test fold, which inflates every score.
- **Preprocessing lives inside a `Pipeline`**, so StandardScaler is re-fitted on each
  training fold.
- **Every model uses the same split objects**, so the scores are genuinely paired.

## 3. Why the naive paired t-test on CV scores is wrong

With 10-fold CV, any two training sets share about 89% of their rows, so the fold
scores are **positively correlated**. The textbook t-test assumes independence:

$$ t_{naive} = \frac{\bar d}{\sqrt{\hat\sigma^2 / J}} $$

When the samples are correlated, $\hat\sigma^2/J$ underestimates the true variance of
$\bar d$, so *t* is too big, *p* is too small, and you "discover" differences that
don't exist. Dietterich (1998) showed Type-I error rates well above the nominal 5%.
Adding more repeats makes it *worse*, because *J* grows while the information doesn't.

## 4. Test 1 (primary): Nadeau–Bengio corrected resampled t-test

Nadeau & Bengio (2003) proposed inflating the variance by a factor that reflects the
train/test overlap. For *r* × *k*-fold CV with *J = r·k* scores (the form recommended by
Bouckaert & Frank, 2004):

$$ t = \frac{\bar d}{\sqrt{\left(\frac{1}{J} + \frac{n_{test}}{n_{train}}\right)\hat\sigma^2}}, \qquad df = J - 1 $$

Here, J = 100 and n_test/n_train ≈ 1083/9751 ≈ 0.111, so the correction factor is about 0.121
instead of 0.01. **The standard error is about 3.5× larger than the naive one.** The
95% CI uses the same corrected standard error.

## 5. Test 2: Dietterich's 5×2cv paired t-test (+ Alpaydin's combined F)

There are five repetitions of 2-fold CV. In each repetition *i*, compute the differences
on both folds $p_i^{(1)}, p_i^{(2)}$, their mean $\bar p_i$, and the variance
$s_i^2 = (p_i^{(1)}-\bar p_i)^2 + (p_i^{(2)}-\bar p_i)^2$.

$$ t = \frac{p_1^{(1)}}{\sqrt{\tfrac{1}{5}\sum_i s_i^2}} \sim t_5 \qquad
   F = \frac{\sum_i\sum_j (p_i^{(j)})^2}{2\sum_i s_i^2} \sim F_{10,5} $$

With 2 folds, the training and test sets in a repetition don't overlap, which keeps the
Type-I error near nominal. The cost is low power: it trains on only 50% of the data,
and the numerator uses a single difference. Alpaydin's combined F-test uses all 10
differences and is more stable. Both are reported.

*Verification:* `tests/test_stats.py::test_5x2cv_matches_mlxtend` reproduces mlxtend's
`paired_ttest_5x2cv` and `combined_ftest_5x2cv` exactly. In the experiment, the splits
are additionally stratified.

## 6. Test 3: McNemar's test on the hold-out set

Fit each model on the full dev set and predict the 2,709 hold-out beans. Then build the table:

|  | Model 2 right | Model 2 wrong |
|---|---|---|
| **Model 1 right** | a | **b** |
| **Model 1 wrong** | **c** | d |

Only the discordant cells *b* and *c* carry information. Under H₀, a disagreement is
equally likely to go either way, so *b* ~ Binomial(*b*+*c*, ½). The **exact** version is used
(always valid). The χ² with continuity correction is also stored in the CSV.

Note that McNemar tests **error rates (accuracy)**, not macro-F1. That's why it's paired
with the bootstrap below.

## 7. Test 4: paired bootstrap on the hold-out set

Resample the 2,709 hold-out indices with replacement 10,000 times. Using the **same
indices** for both models, recompute the macro-F1 difference each time. The 2.5% and 97.5%
percentiles give a 95% CI for the hold-out macro-F1 difference. A vectorised macro-F1
(checked against sklearn in the tests) keeps this fast.

## 8. Test 5: TOST equivalence

A non-significant result does **not** mean "no difference". It may simply be a lack of
power. To claim the models are *practically the same*, the burden of proof has to flip:

- H₀: |Δ| ≥ 0.01 (they differ meaningfully)
- H₁: |Δ| < 0.01 (equivalent within the margin)

Run two one-sided t-tests with the **corrected** standard error. Equivalence holds when
both are significant at α = 0.05, which is the same as the 90% CI lying inside ±0.01.

## 9. Multiple comparisons: Holm–Bonferroni

Three pairwise comparisons at α = 0.05 each would give a family-wise error of up to about 14%.
Holm's step-down method controls it at 5% and is uniformly more powerful than plain
Bonferroni. Sort the p-values, multiply the *i*-th smallest by (m − i + 1), and enforce
monotonicity. The correction is applied within each test family.

## 10. Effect sizes

- **Δ (mean difference)** in macro-F1 points. This is the number that matters for decisions.
- **95% CI** (corrected).
- **Cohen's d_z** = mean(d) / sd(d). This is a descriptive standardised effect. Note that it uses
  the *uncorrected* SD, so it is not comparable to "d" from independent-sample studies.
- **Win/loss count** across the 100 folds.

## 11. Decision rule (pre-registered)

```
if  p_Holm(corrected t) < 0.05  and |Δ| ≥ 0.01   → prefer the better model
elif TOST equivalent within ±0.01                → practically equivalent → choose simpler/cheaper
elif p_Holm < 0.05                               → real but too small to matter
else                                             → inconclusive
```

## 12. Hamming Loss

For a **single-label** multi-class problem:

$$ \text{Hamming Loss} = \frac{1}{n}\sum_i \mathbb{1}[\hat y_i \ne y_i] = 1 - \text{Accuracy} $$

It only differs from 1 − accuracy for **multi-label** problems, where it averages the
errors over every (sample, label) pair. It is reported because the brief asks for it,
and `tests/test_stats.py` checks the identity. It is not treated as independent
evidence.

## References

1. Nadeau, C. & Bengio, Y. (2003). *Inference for the Generalization Error.* Machine Learning, 52, 239–281.
2. Bouckaert, R. R. & Frank, E. (2004). *Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms.* PAKDD 2004, LNCS 3056.
3. Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation, 10(7), 1895–1923.
4. Alpaydin, E. (1999). *Combined 5×2 cv F Test for Comparing Supervised Classification Learning Algorithms.* Neural Computation, 11(8), 1885–1892.
5. McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 12(2), 153–157.
6. Schuirmann, D. J. (1987). *A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability.* J. Pharmacokinetics and Biopharmaceutics, 15(6), 657–680.
7. Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.* Scandinavian Journal of Statistics, 6(2), 65–70.
8. Koklu, M. & Ozkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
