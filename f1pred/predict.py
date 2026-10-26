"""Predict the finishing order of the next race.

Trains on every completed race, then scores the upcoming entry list. The
three models predict a finishing position each; the ensemble is their mean,
and the printed order is the ranking of that mean. Ranking rather than
rounding is what guarantees a real finishing order rather than three drivers
all predicted P4.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd

from f1pred import models
from f1pred.backtest import _training_rows
from f1pred.config import Config
from f1pred.db import read_sql, start_run, upsert_df
from f1pred.features import build_features, next_event
from f1pred.models import MODEL_NAMES


# scripts/tune.py stores the recency setting under this pseudo model name,
# because it is shared by all three models rather than belonging to one.
RECENCY_KEY = "_recency"


def _tuned_rows(mode: str) -> dict[str, dict]:
    df = read_sql(
        "SELECT model_name, params FROM ml.tuned_params WHERE mode = :mode",
        {"mode": mode},
    )
    return {r.model_name: dict(r.params) for _, r in df.iterrows()}


def tuned_overrides(mode: str) -> dict[str, dict]:
    """Per-model hyperparameters found by scripts/tune.py, if it has run."""
    return {k: v for k, v in _tuned_rows(mode).items() if k != RECENCY_KEY}


def apply_tuned_recency(cfg: Config, mode: str) -> tuple[Config, bool]:
    """Use the tuned half-life, which is the setting that matters most.

    Without this the search would find a half-life, store it, and then have
    it quietly ignored at prediction time - the one parameter most worth
    tuning in a regulation-change season.
    """
    tuned = _tuned_rows(mode).get(RECENCY_KEY)
    if not tuned:
        return cfg, False
    raw = copy.deepcopy(cfg.raw)
    raw["training"]["recency_half_life_races"] = tuned["recency_half_life_races"]
    raw["training"]["season_weight_multiplier"] = {
        int(k): float(v)
        for k, v in (tuned.get("season_weight_multiplier") or {}).items()
    }
    return replace(cfg, raw=raw), True


def predict_next(cfg: Config, mode: str, persist: bool = True) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Returns (ordered prediction table, event info, training-set summary)."""
    cfg, used_tuned = apply_tuned_recency(cfg, mode)

    event = next_event()
    if event is None:
        raise RuntimeError(
            "no upcoming race found - the season may be over, or you may "
            "need to run scripts/ingest.py to refresh the calendar"
        )

    features = build_features(include_upcoming=True)
    feature_names = cfg.feature_names(mode)

    test = features[features["event_id"] == int(event["event_id"])].copy()
    if test.empty:
        raise RuntimeError(f"no entry list built for event {event['event_id']}")

    history = features.dropna(subset=["finish_position"])
    train = history[history["event_date"] < event["event_date"]]
    fitting = _training_rows(train, cfg)

    # Ask the raw table, not the feature frame: by this point grid_position
    # has already been imputed to the field median, so it is never NaN and a
    # missing grid would silently look like every car starting 11th.
    if mode == "post_quali":
        have_quali = read_sql(
            "SELECT count(*) AS n FROM raw.qualifying WHERE event_id = :eid",
            {"eid": int(event["event_id"])},
        )["n"].iloc[0]
        if not have_quali:
            raise RuntimeError(
                f"post_quali needs the grid, but no qualifying is stored for "
                f"{event.get('event_name')} yet.\n"
                f"Use --mode pre_weekend, or re-run scripts/ingest.py once "
                f"qualifying has happened."
            )

    X_train = fitting[feature_names]
    y_train = fitting["finish_position"].astype(float)
    weights = models.sample_weights(fitting, cfg)

    out = test[["driver_code", "team_key"]].copy()
    # In pre_weekend mode grid_position holds an imputed field median, which
    # would read as a real starting slot. Blank it rather than mislead.
    out["grid"] = (test["grid_position"].to_numpy() if mode == "post_quali"
                   else np.nan)

    fitted = {}
    for name, model in models.make_models(cfg, tuned_overrides(mode)).items():
        models.fit(model, X_train, y_train, weights)
        out[name] = model.predict(test[feature_names])
        fitted[name] = model

    out["ensemble"] = out[list(MODEL_NAMES)].mean(axis=1)

    # Retirement risk is a separate question from pace, so it gets its own
    # model and its own column rather than being folded into the position.
    dnf_clf = models.make_dnf_classifier()
    dnf_train = train.dropna(subset=["dnf"])
    y_dnf = dnf_train["dnf"].astype(int)
    if y_dnf.nunique() > 1:
        models.fit(dnf_clf, dnf_train[feature_names], y_dnf,
                   models.sample_weights(dnf_train, cfg))
        out["p_dnf"] = dnf_clf.predict_proba(test[feature_names])[:, 1]
    else:
        out["p_dnf"] = np.nan

    out = out.sort_values("ensemble").reset_index(drop=True)
    out.insert(0, "pred", np.arange(1, len(out) + 1))

    info = {
        "event_name": event.get("event_name"),
        "event_id": int(event["event_id"]),
        "season": int(event["season"]),
        "round": int(event["round"]),
        "event_date": event["event_date"],
        "n_train_races": int(fitting["event_id"].nunique()),
        "n_train_rows": int(len(fitting)),
        "mode": mode,
        "tuned": used_tuned,
    }

    if persist:
        run_id = start_run(
            kind="predict", mode=mode,
            config_json={"training": cfg.raw["training"],
                         "features": feature_names,
                         "event_id": info["event_id"]},
            notes=f"next-race prediction, round {info['round']} {info['season']}",
        )
        rows = out.melt(
            id_vars=["driver_code", "pred", "p_dnf"],
            value_vars=list(MODEL_NAMES) + ["ensemble"],
            var_name="model_name", value_name="predicted_value",
        )
        rows["run_id"] = run_id
        rows["event_id"] = info["event_id"]
        rows["predicted_rank"] = rows.groupby("model_name")["predicted_value"] \
            .rank(method="first").astype(int)
        rows["actual_position"] = np.nan
        upsert_df(rows[["run_id", "event_id", "driver_code", "model_name",
                        "predicted_value", "predicted_rank", "p_dnf",
                        "actual_position"]],
                  "ml.predictions",
                  ["run_id", "event_id", "driver_code", "model_name"])

    importance = pd.concat(
        {name: models.importances(m, feature_names) for name, m in fitted.items()},
        axis=1,
    )
    return out, info, importance


def format_prediction(out: pd.DataFrame, info: dict, cfg: Config) -> str:
    name = info["event_name"] or f"round {info['round']}"
    date = pd.to_datetime(info["event_date"]).date()
    lines = [
        f"Next race: {name} - round {info['round']}, {date}",
        f"Mode: {info['mode']}   |   trained on {info['n_train_races']} races "
        f"({info['n_train_rows']} rows), recency half-life "
        f"{cfg.recency_half_life}"
        + ("  [tuned]" if info.get("tuned") else "  [config defaults]"),
        "",
        f"{'Pred':>4}  {'Driver':<6} {'Team':<14}{'Grid':>5}"
        f"{'Ridge':>8}{'RF':>7}{'XGB':>7}{'Ens':>7}{'P(DNF)':>8}",
    ]
    lines.append("-" * len(lines[-1]))
    for _, r in out.iterrows():
        grid = "-" if pd.isna(r["grid"]) else f"{int(r['grid']):d}"
        pdnf = "-" if pd.isna(r["p_dnf"]) else f"{r['p_dnf']:.2f}"
        lines.append(
            f"{int(r['pred']):>4}  {r['driver_code']:<6} {r['team_key']:<14}"
            f"{grid:>5}{r['ridge']:>8.1f}{r['rf']:>7.1f}{r['xgb']:>7.1f}"
            f"{r['ensemble']:>7.1f}{pdnf:>8}"
        )
    return "\n".join(lines)


def format_importance(importance: pd.DataFrame, top: int = 12) -> str:
    lines = ["", f"Top {top} features by importance "
                 "(ridge = |standardised coefficient|, trees = impurity gain)",
             f"{'feature':<34}{'ridge':>9}{'rf':>9}{'xgb':>9}"]
    lines.append("-" * len(lines[-1]))
    ordered = importance.div(importance.sum()).mean(axis=1).sort_values(
        ascending=False).head(top)
    for feat in ordered.index:
        row = importance.loc[feat]
        lines.append(f"{feat:<34}{row.get('ridge', np.nan):>9.3f}"
                     f"{row.get('rf', np.nan):>9.3f}{row.get('xgb', np.nan):>9.3f}")
    return "\n".join(lines)
