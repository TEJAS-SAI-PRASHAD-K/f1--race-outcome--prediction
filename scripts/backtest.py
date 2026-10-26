"""Compare the models against the naive baselines, chronologically.

    python scripts/backtest.py --mode post_quali
    python scripts/backtest.py --mode pre_weekend --races 40
    python scripts/backtest.py --mode post_quali --shuffle-target   # leak probe
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.backtest import run  # noqa: E402
from f1pred.config import MODES, load_config  # noqa: E402
from f1pred.metrics import format_table  # noqa: E402
from f1pred.predict import apply_tuned_recency, tuned_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default="post_quali")
    ap.add_argument("--races", type=int, default=None,
                    help="how many recent races to score")
    ap.add_argument("--shuffle-target", action="store_true",
                    help="scramble the target to prove there is no leakage")
    ap.add_argument("--use-tuned", action="store_true",
                    help="score the settings stored by scripts/tune.py "
                         "instead of the config.yaml defaults")
    args = ap.parse_args()

    cfg = load_config()
    overrides = None
    source = "config.yaml"
    if args.use_tuned:
        cfg, found = apply_tuned_recency(cfg, args.mode)
        overrides = tuned_overrides(args.mode)
        source = "ml.tuned_params" if found else "config.yaml (nothing tuned yet)"

    print(f"mode={args.mode}  half_life={cfg.recency_half_life}  "
          f"features={len(cfg.feature_names(args.mode))}  settings={source}")
    results, _ = run(cfg, args.mode, n_test_races=args.races,
                     model_overrides=overrides,
                     shuffle_target=args.shuffle_target)
    print()
    print(format_table(results))

    if args.shuffle_target:
        print("\nTarget was shuffled. Every model should now sit near the "
              "random baseline (~6.7 MAE). Anything much better is leakage.")


if __name__ == "__main__":
    main()
