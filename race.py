#!/usr/bin/env python3
"""
================================================================================
 ONE FILE TO DRIVE THE WHOLE THING.  Edit the knobs below, then:

     .venv/bin/python race.py

 Everything here overrides config.yaml for this run only, so you can try an
 idea without touching the committed defaults. The modelling itself lives in
 f1pred/ and is deliberately NOT copied in here - see the note at the bottom
 of this comment for why that matters.
================================================================================
"""

# =============================================================================
#  KNOBS
# =============================================================================

# --- what to predict ---------------------------------------------------------
MODE = "pre_weekend"        # "pre_weekend" = no grid yet (run it any day)
                            # "post_quali"  = grid known (run it Saturday night)

# --- which models ------------------------------------------------------------
MODELS = ["rf", "xgb"]      # any of: "ridge", "rf", "xgb"

# --- how much the past counts ------------------------------------------------
# THE MOST IMPORTANT NUMBER HERE. A training row's weight halves every
# RECENCY_HALF_LIFE races going back. 2026 runs new technical regulations, so
# older races describe cars that no longer exist. Low = trust recent form.
RECENCY_HALF_LIFE = 40      # try 3, 10, 40, 100 and watch the backtest move
SEASON_2026_WEIGHT = 1.0    # extra multiplier on 2026 rows, on top of the decay

# --- model hyperparameters ---------------------------------------------------
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,         # deeper = more flexible, more overfitting
    "min_samples_leaf": 5,  # higher = smoother
    "random_state": 42,
    "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,  # lower = slower, usually steadier
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,      # higher = more regularised
    "random_state": 42,
}

# --- what-if: force someone's starting slot ----------------------------------
# post_quali only, because pre_weekend never shows the models a grid position.
# e.g. {"VER": 1, "NOR": 12} to ask "what if Verstappen took pole and Norris
# got a five-place penalty?"
GRID_OVERRIDE: dict[str, int] = {}

# --- output ------------------------------------------------------------------
USE_TUNED = True            # prefer settings found by scripts/tune.py
SHOW_EXPLAIN = True         # feature importances
SHOW_BACKTEST = True        # score the models against the naive baselines
BACKTEST_RACES = 30         # how many recent races to score

# =============================================================================
#  A knob you will notice is MISSING: weather.
#
#  Other F1 scripts expose rain_probability / temperature. Resist copying
#  that. If every training row carries the same constant - and they do, when
#  you assign one number to the whole DataFrame - the column holds no
#  information, the model cannot learn anything from it, and changing the
#  value cannot change the predicted order. It is a dial wired to nothing.
#
#  To make weather real you need per-race historical weather in the training
#  set, which FastF1 moved behind an F1 TV subscription (see README). Until
#  that exists here, there is no honest weather knob to offer.
# =============================================================================


import copy
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from f1pred.backtest import run as backtest_run          # noqa: E402
from f1pred.config import load_config                    # noqa: E402
from f1pred.metrics import format_table                  # noqa: E402
from f1pred.predict import (apply_tuned_recency, format_importance,  # noqa: E402
                            format_prediction, predict_next,
                            tuned_overrides)


def build_config():
    """Apply the knobs above on top of config.yaml, for this run only."""
    cfg = load_config()
    raw = copy.deepcopy(cfg.raw)

    raw["models"]["enabled"] = list(MODELS)
    raw["models"]["rf"]["defaults"] = dict(RF_PARAMS)
    raw["models"]["xgb"]["defaults"] = dict(XGB_PARAMS)
    raw["training"]["recency_half_life_races"] = RECENCY_HALF_LIFE
    raw["training"]["season_weight_multiplier"] = {2026: SEASON_2026_WEIGHT}
    raw["backtest"]["n_test_races"] = BACKTEST_RACES

    cfg = replace(cfg, raw=raw)

    # Tuned settings are stored per mode by scripts/tune.py. They override the
    # recency knob above, which is why this prints which one actually applied.
    overrides = None
    if USE_TUNED:
        cfg, _ = apply_tuned_recency(cfg, MODE)
        overrides = {k: v for k, v in tuned_overrides(MODE).items()
                     if k in MODELS}
    return cfg, overrides


def main() -> None:
    cfg, overrides = build_config()

    out, info, importance = predict_next(cfg, MODE,
                                         grid_overrides=GRID_OVERRIDE)
    print(format_prediction(out, info, cfg))
    if SHOW_EXPLAIN:
        print(format_importance(importance, signed=info.get("ridge_signed")))

    if SHOW_BACKTEST:
        print(f"\nBacktesting on the last {BACKTEST_RACES} races "
              f"(walk-forward, chronological)...")
        results, _ = backtest_run(cfg, MODE, n_test_races=BACKTEST_RACES,
                                  model_overrides=overrides, verbose=False)
        print()
        print(format_table(results))
        print("\nRead this table against the baselines, not on its own. A model "
              "\nthat cannot beat baseline_grid_order is not adding anything.")


if __name__ == "__main__":
    main()
