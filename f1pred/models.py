"""The three regressors, behind one interface.

Everything downstream (backtest, tune, predict) treats a model as an opaque
thing with fit/predict, so adding a fourth algorithm never means touching
those files.

A note on model choice at this data size: there are roughly 3,800 rows, and
recency weighting shrinks the effective count further. That is small. Ridge
is a serious contender, not a warm-up act, and the tree models are
deliberately shallow and heavily regularised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from f1pred.config import Config

MODEL_NAMES = ("ridge", "rf", "xgb")


def make_model(name: str, params: dict):
    """Build one estimator. `params` comes from config.yaml or ml.tuned_params."""
    if name == "ridge":
        # Scaling is mandatory, not cosmetic: Ridge penalises raw coefficient
        # size, so without it a feature measured in laps is punished far more
        # than one measured in percent. It also makes the coefficients
        # directly readable as feature importance.
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(**params)),
        ])
    if name == "rf":
        return RandomForestRegressor(**params)
    if name == "xgb":
        return XGBRegressor(**params)
    raise ValueError(f"unknown model {name!r}")


def make_models(cfg: Config, overrides: dict[str, dict] | None = None) -> dict:
    overrides = overrides or {}
    return {
        name: make_model(name, {**cfg.model_defaults(name),
                                **overrides.get(name, {})})
        for name in MODEL_NAMES
    }


def make_dnf_classifier() -> Pipeline:
    """P(this car does not finish).

    Kept separate from the position model on purpose. Retirements are a
    different process from raw pace, and folding them into one number would
    hide which of the two the model is actually uncertain about.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def make_dnf_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        random_state=42, n_jobs=-1,
    )


# --------------------------------------------------------------------------
# recency weighting - the knob that matters most in 2026
# --------------------------------------------------------------------------

def sample_weights(train: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Weight each training row by how recent its race is.

    2026 is year one of new technical regulations, so a 2019 race says very
    little about the current pecking order. An exponential decay lets old
    seasons keep teaching circuit and driver effects while current form
    dominates.

        weight = 0.5 ** (races_ago / half_life) * season_multiplier
    """
    races = (train[["event_id", "event_date"]]
             .drop_duplicates()
             .sort_values(["event_date", "event_id"])
             .reset_index(drop=True))
    races["race_index"] = np.arange(len(races))
    latest = races["race_index"].max()

    idx = train[["event_id"]].merge(races[["event_id", "race_index"]],
                                    on="event_id", how="left")["race_index"]
    races_ago = (latest - idx).to_numpy(dtype=float)

    half_life = max(cfg.recency_half_life, 1e-6)
    weights = 0.5 ** (races_ago / half_life)

    multipliers = cfg.season_weight_multiplier
    if multipliers:
        season_mult = train["season"].map(multipliers).fillna(1.0).to_numpy(float)
        weights = weights * season_mult

    return weights


def fit(model, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None = None):
    """Fit, passing sample weights through whatever wrapper is in the way."""
    if weights is None:
        model.fit(X, y)
    elif isinstance(model, Pipeline):
        # Pipelines route fit params with a "<step>__" prefix.
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": weights})
    else:
        model.fit(X, y, sample_weight=weights)
    return model


# --------------------------------------------------------------------------
# explainability, with no extra dependencies
# --------------------------------------------------------------------------

def importances(model, feature_names: list[str]) -> pd.Series:
    """One importance score per feature, comparable within a model.

    Ridge: absolute standardised coefficients (a one-standard-deviation move
    in the feature shifts the prediction by this many positions).
    Trees: the library's own impurity-based importance.
    """
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    elif hasattr(estimator, "feature_importances_"):
        values = np.ravel(estimator.feature_importances_)
    else:
        return pd.Series(dtype=float)

    return pd.Series(values, index=feature_names).sort_values(ascending=False)
