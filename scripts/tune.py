"""Search recency and model hyperparameters, then store the winners.

    python scripts/tune.py --mode post_quali

Results land in ml.tuned_params and are picked up automatically by
scripts/predict_next.py. Every trial is kept in ml.backtest_metrics, so the
whole search can be queried afterwards.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.config import MODES, load_config  # noqa: E402
from f1pred.tune import tune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default="post_quali")
    args = ap.parse_args()

    best = tune(load_config(), args.mode)
    print("=" * 60)
    print(f"best recency : half_life={best['half_life']}, "
          f"2026 multiplier={best['season_multiplier']}")
    for name, params in best["params"].items():
        print(f"best {name:<6}: {params}  (MAE {best['mae'][name]:.3f})")
    print(f"\nstored in ml.tuned_params (run {best['run_id']}). "
          f"Copy the recency values into config.yaml to make them the default.")


if __name__ == "__main__":
    main()
