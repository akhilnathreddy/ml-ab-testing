# ML A/B Testing: Logistic Regression vs Random Forest vs LightGBM

> 🚧 Work in progress. See [`docs/PLAN.md`](docs/PLAN.md) for the pre-registered design and
> [`docs/WORKLOG.md`](docs/WORKLOG.md) for progress.

This is an end-to-end, statistically rigorous A/B test of three classifiers on the
**Dry Bean** multi-class benchmark. It uses macro-F1, Precision, Recall, Hamming Loss and
more, and compares the models with a corrected resampled t-test, the 5×2cv t-test,
McNemar's test, a paired bootstrap and TOST equivalence.
