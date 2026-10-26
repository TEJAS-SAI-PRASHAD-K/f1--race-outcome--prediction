"""Print the predicted finishing order for the next race.

    python scripts/predict_next.py --mode pre_weekend
    python scripts/predict_next.py --mode post_quali --explain
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1pred.config import MODES, load_config  # noqa: E402
from f1pred.predict import (apply_tuned_recency, format_importance,  # noqa: E402
                            format_prediction, predict_next)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default="pre_weekend")
    ap.add_argument("--explain", action="store_true",
                    help="also print feature importances")
    args = ap.parse_args()

    cfg = load_config()
    out, info, importance = predict_next(cfg, args.mode)
    # Report the half-life actually used, which tuning may have overridden.
    effective_cfg, _ = apply_tuned_recency(cfg, args.mode)
    print(format_prediction(out, info, effective_cfg))
    if args.explain:
        print(format_importance(importance))


if __name__ == "__main__":
    main()
