"""Run the full A/B testing experiment and write all tables + figures to results/.

Usage:
    python scripts/run_experiment.py            # full run (about 10 min on 4 cores)
    python scripts/run_experiment.py --no-plots
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abtest import config  # noqa: E402
from abtest.experiment import run, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--n-boot", type=int, default=config.N_BOOTSTRAP)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    res = run(n_jobs=args.n_jobs, n_boot=args.n_boot)
    save(res)
    # Cache the full results object so figures/notebook can be rebuilt without retraining.
    with open(config.DATA_DIR / "results_cache.pkl", "wb") as fh:
        pickle.dump(res, fh)

    if not args.no_plots:
        from abtest.plots import make_all_figures

        make_all_figures(res)

    print(f"\nDone in {res.meta['runtime_seconds']} s. Results in {config.RESULTS_DIR}")
    print(res.decisions.to_string(index=False))


if __name__ == "__main__":
    main()
