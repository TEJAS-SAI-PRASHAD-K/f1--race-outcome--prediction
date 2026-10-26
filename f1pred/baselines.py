"""Naive predictors that any real model has to beat.

This is the most important file for keeping yourself honest. "MAE 3.0" sounds
fine until you learn that simply predicting everyone finishes where they
started scores 3.2 at the same track. Every backtest prints these alongside
the models, and a model that cannot clear them is not adding anything.
"""

from __future__ import annotations

import pandas as pd

# name -> (column to rank on, ascending, which modes it is valid in)
BASELINES: dict[str, tuple[str, bool, tuple[str, ...]]] = {
    # The bar. Grid position alone is a very strong predictor.
    "baseline_grid_order": ("grid_position", True, ("post_quali",)),
    # Where the driver finished last time out.
    "baseline_last_race": ("prev_finish_position", True,
                           ("pre_weekend", "post_quali")),
    # Championship order going into the weekend - points banked so far,
    # descending, so the leader is predicted to win.
    "baseline_championship": ("season_points_to_date", False,
                              ("pre_weekend", "post_quali")),
}


def applicable(mode: str) -> list[str]:
    return [n for n, (_, _, modes) in BASELINES.items() if mode in modes]


def predict(df: pd.DataFrame, name: str) -> pd.Series:
    """Rankable score for one baseline. Lower always means 'finishes higher'."""
    col, ascending, _ = BASELINES[name]
    values = df[col].astype(float)
    # Flip descending scores so the ranking convention stays consistent.
    return values if ascending else -values
