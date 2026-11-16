"""Figures for the report.

Palette: the first three slots of a validated categorical palette (blue, orange,
aqua), which passes all-pairs CVD and normal-vision separation checks. Aqua sits
below 3:1 contrast on the light surface, so every chart carries direct labels and
the README repeats the numbers as tables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from . import config  # noqa: E402
from .metrics import METRICS  # noqa: E402

# ----------------------------------------------------------------------- tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BAND = "#f0efec"
MODEL_COLORS = {"LogReg": "#2a78d6", "RandomForest": "#eb6834", "LightGBM": "#1baf7a"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": INK,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)


def _signed(v: float, nd: int = 4) -> str:
    """Signed fixed-point without a misleading '-0.0000'."""
    return f"{0.0 if abs(v) < 0.5 * 10**-nd else v:+.{nd}f}"


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom")


# ----------------------------------------------------------------------- 1 class balance
def fig_class_distribution(res) -> None:
    ds = res.dataset
    y = np.concatenate([ds.y_dev, ds.y_holdout])
    counts = pd.Series(y).value_counts().sort_values()
    names = [ds.classes[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(names, counts.values, color=MODEL_COLORS["LogReg"], height=0.62)
    for i, v in enumerate(counts.values):
        ax.text(v + 40, i, f"{v:,}  ({v / len(y):.1%})", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, counts.max() * 1.28)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rows (after removing duplicates)")
    ax.set_title("Dry Bean: 7 classes, moderately imbalanced", pad=22)
    _subtitle(ax, f"{len(y):,} rows · largest class is {counts.max() / counts.min():.1f}× the smallest")
    _save(fig, "01_class_distribution.png")


# ----------------------------------------------------------------------- 2 CV scores
def fig_cv_scores(res) -> None:
    cv = res.cv
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rng = np.random.default_rng(0)
    for i, m in enumerate(config.MODEL_ORDER):
        s = cv.loc[cv.model == m, "f1_macro"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, s.size)
        ax.scatter(s, i + jitter, s=14, color=MODEL_COLORS[m], alpha=0.55, linewidths=0)
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([s.mean()], [i], marker="o", ms=8, color=SURFACE, mec=INK, mew=2, zorder=4)
        ax.text(s.max() + 0.0012, i, f"mean {s.mean():.4f}\nsd {s.std(ddof=1):.4f}",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(3), config.MODEL_ORDER)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo, hi = cv["f1_macro"].min(), cv["f1_macro"].max()
    ax.set_xlim(lo - 0.003, hi + 0.012)
    ax.set_xlabel("Macro F1 on the test fold")
    ax.set_title("100 test folds per model (10×10 repeated stratified CV)", pad=22)
    _subtitle(ax, "dots = folds · bar = interquartile range · ring = mean. Spread across folds is far larger than gaps between models.")
    _save(fig, "02_cv_macro_f1_by_model.png")


# ----------------------------------------------------------------------- 3 forest plot
def fig_forest(res) -> None:
    t = res.cv_tests
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(-config.MPD, config.MPD, color=BAND, zorder=0)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    for i, r in t.reset_index(drop=True).iterrows():
        ax.plot([r.naive_ci_low, r.naive_ci_high], [i - 0.15] * 2, color=MUTED, lw=2,
                solid_capstyle="round")
        ax.plot([r.ci_low, r.ci_high], [i + 0.15] * 2, color=INK, lw=2.5, solid_capstyle="round")
        ax.plot([r.mean_diff], [i + 0.15], "o", ms=8, color=INK, zorder=3)
        ax.plot([r.mean_diff], [i - 0.15], "o", ms=6, color=MUTED, zorder=3)
        ax.text(max(r.ci_high, r.naive_ci_high, config.MPD) + 0.0008, i,
                f"Δ = {r.mean_diff:+.4f}\np(Holm) = {r.p_holm_corrected:.3f}",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(t)), t["comparison"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lim = max(abs(t[["ci_low", "ci_high"]]).to_numpy().max(), config.MPD) * 1.25
    ax.set_xlim(-lim, lim * 1.9)
    ax.set_xlabel("Difference in macro F1 (model 1 − model 2)")
    ax.plot([], [], color=INK, lw=2.5, label="95% CI, corrected (Nadeau–Bengio)")
    ax.plot([], [], color=MUTED, lw=2, label="95% CI, naive paired t-test (too narrow)")
    ax.fill_between([], [], color=BAND, label=f"±{config.MPD} practical-equivalence band")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    ax.set_title("Effect sizes with confidence intervals", pad=22)
    _subtitle(ax, "The naive CI ignores the overlap between training folds, so it understates the uncertainty.")
    _save(fig, "03_forest_plot_differences.png")


# ----------------------------------------------------------------------- 4 p-values
def fig_pvalues(res) -> None:
    t = res.cv_tests
    tests = [
        ("p_holm_naive", "Naive paired t (invalid)", MUTED, "s"),
        ("p_holm_corrected", "Corrected resampled t (primary)", INK, "o"),
        ("p_holm_5x2", "5×2cv paired t", INK_2, "D"),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (col, label, color, marker) in enumerate(tests):
        y = np.arange(len(t)) + (k - 1) * 0.22
        ax.scatter(t[col].clip(lower=1e-16), y, color=color, marker=marker, s=46, zorder=3,
                   label=label)
    ax.axvline(config.ALPHA, color="#d03b3b", lw=1.2, ls="--")
    ax.text(config.ALPHA * 0.85, len(t) - 0.62, "α = 0.05", color="#d03b3b", fontsize=8.5,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(range(len(t)), t["comparison"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Holm-adjusted p-value (log scale; left of the dashed line = 'significant')")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=8.5)
    ax.set_title("Same data, different tests, different conclusions", pad=22)
    _subtitle(ax, "The naive test reports much smaller p-values than the tests that correct for fold dependence.")
    _save(fig, "04_pvalues_by_test.png")


# ----------------------------------------------------------------------- 5 diff histograms
def fig_fold_differences(res) -> None:
    cv = res.cv
    wide = cv.pivot_table(index=["repeat", "fold"], columns="model", values="f1_macro")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    lim = 0
    for (m1, m2) in config.COMPARISONS:
        lim = max(lim, (wide[m1] - wide[m2]).abs().max())
    bins = np.linspace(-lim * 1.05, lim * 1.05, 31)
    for ax, (m1, m2) in zip(axes, config.COMPARISONS):
        d = wide[m1] - wide[m2]
        ax.axvspan(-config.MPD, config.MPD, color=BAND, zorder=0)
        ax.hist(d, bins=bins, color=MODEL_COLORS[m1], edgecolor=SURFACE, linewidth=1.5)
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(d.mean(), color=INK, lw=2)
        ax.set_title(f"{m1} − {m2}", fontsize=10.5)
        ax.text(0.02, 0.95, f"mean {d.mean():+.4f}\n{(d > 0).sum()} wins / {(d < 0).sum()} losses",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
        ax.set_xlabel("Δ macro F1 per fold")
        ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Folds")
    fig.suptitle("Paired per-fold differences (100 folds). Black line = mean, shaded = ±MPD band",
                 x=0.06, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_paired_fold_differences.png")


# ----------------------------------------------------------------------- 6 all metrics
def fig_all_metrics(res) -> None:
    """Small multiples: one panel per metric, each on its own scale."""
    cv = res.cv
    mean = cv.groupby("model")[list(METRICS)].mean().loc[config.MODEL_ORDER]
    sd = cv.groupby("model")[list(METRICS)].std().loc[config.MODEL_ORDER]
    n = np.sqrt(len(cv) / len(config.MODEL_ORDER))
    metrics = list(METRICS)
    ncols = 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.15 * nrows))
    axes = axes.ravel()
    for ax, m in zip(axes, metrics):
        label, higher = METRICS[m]
        vals, err = mean[m], sd[m] / n  # mean ± standard error across folds (descriptive)
        best = vals.idxmax() if higher else vals.idxmin()
        for i, model in enumerate(config.MODEL_ORDER):
            ax.plot([vals[model] - err[model], vals[model] + err[model]], [i, i],
                    color=MODEL_COLORS[model], lw=2, alpha=0.6, solid_capstyle="round")
            ax.scatter(vals[model], i, s=50, color=MODEL_COLORS[model], zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
            ax.text(1.04, i, f"{vals[model]:.4f}" + (" ★" if model == best else ""),
                    transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.5,
                    color=INK if model == best else INK_2,
                    fontweight="bold" if model == best else "normal")
        ax.set_yticks(range(3), config.MODEL_ORDER if ax in axes[::ncols] else [])
        ax.set_ylim(2.6, -0.6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(f"{label}  ({'↑' if higher else '↓'} better)", fontsize=10)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle("Every metric, mean over 100 CV folds (bar = ± standard error, ★ = best). "
                 "Note: each panel has its own x-scale.",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=5)
    _save(fig, "06_all_metrics_cv_mean.png")


# ----------------------------------------------------------------------- 7 confusion
def fig_confusion(res) -> None:
    ds = res.dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, m in zip(axes, config.MODEL_ORDER):
        cm = confusion_matrix(ds.y_holdout, res.holdout_preds[m], normalize="true")
        ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                            color=SURFACE if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(ds.classes)), ds.classes, rotation=45, ha="right")
        ax.set_yticks(range(len(ds.classes)), ds.classes if m == config.MODEL_ORDER[0] else [])
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_title(f"{m} (hold-out macro F1 {res.holdout_metrics.loc[m, 'f1_macro']:.4f})",
                     fontsize=10.5)
        ax.set_xlabel("Predicted")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("True class")
    fig.suptitle("Hold-out confusion matrices (row-normalised = recall per class)", x=0.02,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "07_confusion_matrices_holdout.png")


# ----------------------------------------------------------------------- 8 per-class F1
def fig_per_class(res) -> None:
    ds = res.dataset
    f1 = pd.DataFrame(
        {m: f1_score(ds.y_holdout, res.holdout_preds[m], average=None) for m in config.MODEL_ORDER},
        index=ds.classes,
    ).sort_values("LogReg")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    offsets = dict(zip(config.MODEL_ORDER, [0.22, 0.0, -0.22]))  # legend order, top→bottom
    for j, c in enumerate(f1.index):
        if j % 2 == 0:
            ax.axhspan(j - 0.45, j + 0.45, color=BAND, zorder=0, lw=0)
        for m in config.MODEL_ORDER:
            ax.plot([0, f1.loc[c, m]], [j + offsets[m]] * 2, color=GRID, lw=1, zorder=1)
            ax.scatter(f1.loc[c, m], j + offsets[m], s=42, color=MODEL_COLORS[m], zorder=3,
                       edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(range(len(f1)), f1.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(f1.to_numpy().min() - 0.01, 1.005)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 on the hold-out set")
    for m in config.MODEL_ORDER:
        ax.scatter([], [], s=60, color=MODEL_COLORS[m], label=m)
    ax.legend(loc="lower right", framealpha=1, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.set_title("Per-class F1: the hard classes are hard for every model", pad=22)
    _subtitle(ax, "Most errors are SIRA↔DERMASON, then BARBUNYA↔CALI. BOMBAY is perfectly separated by every model.")
    _save(fig, "08_per_class_f1_holdout.png")


# ----------------------------------------------------------------------- 9 bootstrap
def fig_bootstrap(res) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    ht = res.holdout_tests.set_index("comparison")
    allv = np.concatenate(list(res.bootstrap_dists.values()))
    lim = max(abs(allv.min()), abs(allv.max()), config.MPD) * 1.05
    bins = np.linspace(-lim, lim, 50)
    for ax, (m1, m2) in zip(axes, config.COMPARISONS):
        key = f"{m1} vs {m2}"
        d = res.bootstrap_dists[key]
        r = ht.loc[key]
        ax.axvspan(-config.MPD, config.MPD, color=BAND, zorder=0)
        ax.hist(d, bins=bins, color=MODEL_COLORS[m1], edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(r.boot_ci_low, color=INK, lw=1.2, ls="--")
        ax.axvline(r.boot_ci_high, color=INK, lw=1.2, ls="--")
        ax.set_title(f"{m1} − {m2}\nΔ {_signed(r.macro_f1_diff)}, 95% CI [{_signed(r.boot_ci_low)}, {_signed(r.boot_ci_high)}]",
                     fontsize=10)
        ax.set_xlabel("Δ macro F1 (hold-out)")
        ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Bootstrap resamples")
    fig.suptitle(f"Paired bootstrap on the hold-out set ({res.meta['n_bootstrap']:,} resamples). Dashed = 95% CI",
                 x=0.06, ha="left", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "09_bootstrap_holdout.png")


# ----------------------------------------------------------------------- 10 McNemar
def fig_mcnemar(res) -> None:
    ht = res.holdout_tests
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, r in ht.reset_index(drop=True).iterrows():
        ax.barh(i - 0.18, r.only_1_correct, height=0.34, color=MODEL_COLORS[r.model_1])
        ax.barh(i + 0.18, r.only_2_correct, height=0.34, color=MODEL_COLORS[r.model_2])
        ax.text(r.only_1_correct + 1, i - 0.18, f"only {r.model_1} right: {r.only_1_correct}",
                va="center", fontsize=8.5, color=INK_2)
        ax.text(r.only_2_correct + 1, i + 0.18, f"only {r.model_2} right: {r.only_2_correct}",
                va="center", fontsize=8.5, color=INK_2)
    m = max(ht.only_1_correct.max(), ht.only_2_correct.max())
    ax.set_xlim(0, m * 1.75)
    ax.set_yticks(range(len(ht)), [f"{r.comparison}\nexact p = {r.p_mcnemar:.3f}"
                                   for r in ht.itertuples()])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hold-out samples (discordant pairs only)")
    ax.set_title("McNemar's test: where do the models disagree?", pad=22)
    _subtitle(ax, f"Out of {len(res.dataset.y_holdout):,} hold-out beans. Only the disagreements carry information about which model is better.")
    _save(fig, "10_mcnemar_discordant_pairs.png")


def make_all_figures(res) -> None:
    for fn in [fig_class_distribution, fig_cv_scores, fig_forest, fig_pvalues,
               fig_fold_differences, fig_all_metrics, fig_confusion, fig_per_class,
               fig_bootstrap, fig_mcnemar]:
        fn(res)
    print(f"Figures written to {config.FIGURES_DIR}")
