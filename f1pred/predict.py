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
from f1pred.backtest import training_rows
from f1pred.config import Config
from f1pred.db import read_sql, start_run, upsert_df
from f1pred.features import build_features, next_event


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


def apply_grid_overrides(test: pd.DataFrame, overrides: dict[str, float],
                         mode: str) -> dict[str, float]:
    """What-if: force a driver's starting slot and re-derive what follows.

    Only meaningful in post_quali, where grid_position is the dominant
    feature. In pre_weekend the models never see the grid, so an override
    there would silently do nothing - hence the explicit refusal rather than
    an inert knob.

    Mutates `test` in place. Returns the overrides actually applied.
    """
    if not overrides:
        return {}
    if mode != "post_quali":
        raise ValueError(
            "grid overrides only affect post_quali mode; in pre_weekend the "
            "models are not given grid_position at all, so setting it would "
            "change nothing"
        )

    known = set(test["driver_code"])
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"not on the entry list: {sorted(unknown)}")

    for code, slot in overrides.items():
        row = test["driver_code"] == code
        test.loc[row, "grid_position"] = float(slot)

    # quali_gap_to_pole_pct is a pace measurement, not a function of the
    # slot, so it is deliberately left as measured. Moving a car on the grid
    # does not make it faster.
    return {k: float(v) for k, v in overrides.items()}


def predict_next(cfg: Config, mode: str, persist: bool = True,
                 grid_overrides: dict[str, float] | None = None
                 ) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Returns (ordered prediction table, event info, importances)."""
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
    fitting = training_rows(train, cfg)

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

    applied = apply_grid_overrides(test, grid_overrides or {}, mode)

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

    out["ensemble"] = out[cfg.enabled_models].mean(axis=1)

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
        "grid_overrides": applied,
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
            value_vars=list(cfg.enabled_models) + ["ensemble"],
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
    # Signed Ridge coefficients carry the direction that |value| throws away.
    # Only available when ridge is one of the enabled models.
    if "ridge" in fitted:
        coefs = fitted["ridge"].named_steps["model"].coef_
        info["ridge_signed"] = pd.Series(np.ravel(coefs), index=feature_names)
    return out, info, importance


def format_prediction(out: pd.DataFrame, info: dict, cfg: Config) -> str:
    name = info["event_name"] or f"round {info['round']}"
    date = pd.to_datetime(info["event_date"]).date()
    model_cols = [m for m in cfg.enabled_models if m in out.columns]

    header = (f"{'Pred':>4}  {'Driver':<6} {'Team':<14}{'Grid':>5}"
              + "".join(f"{m.upper():>8}" for m in model_cols)
              + f"{'Ens':>7}{'P(DNF)':>8}")
    lines = [
        f"Next race: {name} - round {info['round']}, {date}",
        f"Mode: {info['mode']}   |   trained on {info['n_train_races']} races "
        f"({info['n_train_rows']} rows), recency half-life "
        f"{cfg.recency_half_life}"
        + ("  [tuned]" if info.get("tuned") else "  [config defaults]"),
        f"Models: {', '.join(model_cols)}"
        + (f"   |   grid overrides: {info['grid_overrides']}"
           if info.get("grid_overrides") else ""),
        "",
        header,
        "-" * len(header),
    ]
    for _, r in out.iterrows():
        grid = "-" if pd.isna(r["grid"]) else f"{int(r['grid']):d}"
        pdnf = "-" if pd.isna(r["p_dnf"]) else f"{r['p_dnf']:.2f}"
        lines.append(
            f"{int(r['pred']):>4}  {r['driver_code']:<6} {r['team_key']:<14}"
            f"{grid:>5}"
            + "".join(f"{r[m]:>8.1f}" for m in model_cols)
            + f"{r['ensemble']:>7.1f}{pdnf:>8}"
        )
    return "\n".join(lines)


def format_importance(importance: pd.DataFrame, signed: pd.Series | None = None,
                      top: int = 12) -> str:
    """Importances as each model's percentage share of its own total.

    Raw values are not comparable across models - a Ridge coefficient is in
    positions-per-standard-deviation, a tree importance is a fraction of
    impurity reduction. Normalising each column to 100% makes the columns
    mean the same thing, and the table is sorted on what it prints.

    `signed` carries the Ridge coefficient WITH its sign, which is where the
    direction lives: positive pushes the predicted position number up (a
    worse finish), negative pulls it down.
    """
    cols = list(importance.columns)
    share = importance.div(importance.sum()) * 100.0
    order = share.mean(axis=1).sort_values(ascending=False).head(top)

    head = f"{'feature':<34}" + "".join(f"{c + ' %':>9}" for c in cols)
    if signed is not None:
        head += f"{'ridge coef':>12}"
    lines = ["", f"Top {top} features - each column is that model's % of its "
                 "own total", head, "-" * len(head)]
    for feat in order.index:
        row = share.loc[feat]
        line = f"{feat:<34}" + "".join(f"{row.get(c, np.nan):>9.1f}" for c in cols)
        if signed is not None:
            value = signed.get(feat)
            line += "         -  " if value is None or pd.isna(value) \
                else f"{value:>12.3f}"
        lines.append(line)
    if signed is not None:
        lines.append("")
        lines.append("ridge coef: change in predicted finishing position per "
                     "1 SD of the feature")
        lines.append("            (+ = finishes further back, - = further up)")
    return "\n".join(lines)
